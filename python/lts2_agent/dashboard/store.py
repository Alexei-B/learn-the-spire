"""Run store: read training-run event files, incrementally, and serve queries.

The store owns all file access for the dashboard. It is deliberately dumb about
the *meaning* of metrics — names and tag values are opaque strings, grouped
dynamically. Two jobs:

* **Incremental tail.** Each run's ``events.jsonl`` grows while the run is live.
  We remember a byte offset per run and, on refresh, read only the appended
  bytes. A final line still mid-write (no trailing newline yet) is held back in a
  ``partial`` buffer until its newline arrives, so a truncated tail never crashes
  a parse and never gets counted twice.
* **Downsampling.** :func:`bucket_series` groups events by ``step // bucket`` and
  optionally by a tag value, returning the mean value and the sample count ``n``
  per bucket — the count matters because a 0.6 win-rate over 5 fights must read
  as thin as it is.

Everything here is stdlib-only and thread-safe (a single lock guards the caches),
which is all a ThreadingHTTPServer for a handful of local runs needs.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Optional

# Read tail chunk size for the cheap /api/runs lastEventTs probe.
_TAIL_BYTES = 8192


class _RunCache:
    """Parsed, incrementally-updated state for one run directory."""

    def __init__(self, run_id: str, run_dir: str) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.events_path = os.path.join(run_dir, "events.jsonl")
        self.manifest_path = os.path.join(run_dir, "manifest.json")

        # Incremental-tail bookkeeping.
        self.offset = 0                       # bytes of events.jsonl already read
        self.partial = b""                    # trailing incomplete-line bytes

        # Parsed events, kept as parallel to what queries need.
        self.events: list[dict[str, Any]] = []
        self.metrics: set[str] = set()
        self.tag_keys: dict[str, set[str]] = {}
        self.max_step = 0
        self.last_ts = 0.0

        # Manifest cache, invalidated by mtime.
        self._manifest: dict[str, Any] = {}
        self._manifest_mtime: float = -1.0

    def reset(self) -> None:
        """Forget everything parsed (used when a file shrinks / is rotated)."""
        self.offset = 0
        self.partial = b""
        self.events.clear()
        self.metrics.clear()
        self.tag_keys.clear()
        self.max_step = 0
        self.last_ts = 0.0

    # -- manifest -----------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        try:
            mtime = os.path.getmtime(self.manifest_path)
        except OSError:
            return self._manifest
        if mtime != self._manifest_mtime:
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._manifest = loaded
            except (OSError, ValueError):
                pass  # keep the last good manifest; a half-written file is transient
            self._manifest_mtime = mtime
        return self._manifest

    # -- incremental parse --------------------------------------------------
    def refresh(self) -> None:
        """Read any bytes appended since the last refresh and parse full lines."""
        try:
            size = os.path.getsize(self.events_path)
        except OSError:
            return  # events.jsonl not created yet
        if size < self.offset:
            # File shrank (rotation / restart): re-read from scratch.
            self.reset()
        if size == self.offset:
            return
        try:
            with open(self.events_path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return
        self.offset += len(chunk)
        buf = self.partial + chunk
        parts = buf.split(b"\n")
        # The last element is whatever follows the final newline: an incomplete
        # line still being written (or b"" when the chunk ended cleanly). Hold it
        # back so a truncated tail is parsed only once its newline lands.
        self.partial = parts.pop()
        for raw in parts:
            self._ingest(raw)

    def _ingest(self, raw: bytes) -> None:
        line = raw.strip()
        if not line:
            return
        try:
            ev = json.loads(line)
        except ValueError:
            return  # tolerate a stray corrupt line rather than dropping the run
        if not isinstance(ev, dict) or "name" not in ev:
            return
        name = ev.get("name")
        if not isinstance(name, str):
            return
        try:
            step = int(ev.get("step", 0))
        except (TypeError, ValueError):
            step = 0
        try:
            value = float(ev.get("value", 0.0))
        except (TypeError, ValueError):
            return
        tags = ev.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        try:
            ts = float(ev.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0

        self.events.append({"name": name, "step": step, "value": value, "tags": tags})
        self.metrics.add(name)
        if tags:
            keys = self.tag_keys.setdefault(name, set())
            keys.update(tags.keys())
        else:
            self.tag_keys.setdefault(name, set())
        if step > self.max_step:
            self.max_step = step
        if ts > self.last_ts:
            self.last_ts = ts


def _tail_last_ts(path: str, size: int) -> float:
    """Cheaply read the last complete line's ``ts`` without parsing the file."""
    if size <= 0:
        return 0.0
    start = max(0, size - _TAIL_BYTES)
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            data = fh.read()
    except OSError:
        return 0.0
    lines = [ln for ln in data.split(b"\n") if ln.strip()]
    # Walk backwards; the very last line may be a mid-write fragment.
    for raw in reversed(lines):
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        if isinstance(ev, dict) and "ts" in ev:
            try:
                return float(ev["ts"])
            except (TypeError, ValueError):
                continue
    return 0.0


def bucket_series(
    events: list[dict[str, Any]],
    name: str,
    group_by: Optional[str] = None,
    bucket: Any = "auto",
    target_points: int = 500,
) -> dict[str, list[dict[str, Any]]]:
    """Group ``events`` of one metric into per-group downsampled point lists.

    * ``group_by`` — ``None``/``"none"`` puts everything in group ``"all"``;
      otherwise it is a tag key and each distinct tag value is its own group
      (events missing that tag fall in ``"(untagged)"``).
    * ``bucket`` — an int step-width, or ``"auto"`` to target ``target_points``
      points per group across the metric's step span.

    Each point is ``{"step", "value" (mean over the bucket), "n" (sample count)}``.
    """
    rows = [e for e in events if e["name"] == name]
    grouped = group_by not in (None, "none", "")

    if not rows:
        return {}

    min_step = min(e["step"] for e in rows)
    max_step = max(e["step"] for e in rows)

    width = _resolve_bucket(bucket, min_step, max_step, target_points)

    # (group, bucket_index) -> accumulator
    acc: dict[tuple, dict[str, float]] = {}
    for e in rows:
        if grouped:
            gval = e["tags"].get(group_by, "(untagged)")
        else:
            gval = "all"
        bidx = e["step"] // width
        key = (gval, bidx)
        cell = acc.get(key)
        if cell is None:
            cell = {"sum_v": 0.0, "sum_s": 0.0, "n": 0}
            acc[key] = cell
        cell["sum_v"] += e["value"]
        cell["sum_s"] += e["step"]
        cell["n"] += 1

    groups: dict[str, list[dict[str, Any]]] = {}
    for (gval, _bidx), cell in acc.items():
        n = int(cell["n"])
        point = {
            "step": int(round(cell["sum_s"] / n)),
            "value": cell["sum_v"] / n,
            "n": n,
        }
        groups.setdefault(gval, []).append(point)
    for pts in groups.values():
        pts.sort(key=lambda p: p["step"])
    return groups


def _resolve_bucket(bucket: Any, min_step: int, max_step: int, target_points: int) -> int:
    if bucket == "auto" or bucket is None:
        span = max_step - min_step + 1
        return max(1, math.ceil(span / max(1, target_points)))
    try:
        width = int(bucket)
    except (TypeError, ValueError):
        width = 1
    return max(1, width)


class RunStore:
    """Thread-safe cache of all run directories under ``root``."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._caches: dict[str, _RunCache] = {}
        self._lock = threading.Lock()

    # -- discovery ----------------------------------------------------------
    def _list_run_dirs(self) -> list[tuple[str, str]]:
        """Return ``(run_id, run_dir)`` for every subdir that looks like a run."""
        out: list[tuple[str, str]] = []
        try:
            entries = os.scandir(self.root)
        except OSError:
            return out
        with entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                run_dir = entry.path
                has_events = os.path.exists(os.path.join(run_dir, "events.jsonl"))
                has_manifest = os.path.exists(os.path.join(run_dir, "manifest.json"))
                if has_events or has_manifest:
                    out.append((entry.name, run_dir))
        return out

    def _cache_for(self, run_id: str, run_dir: str) -> _RunCache:
        rc = self._caches.get(run_id)
        if rc is None:
            rc = _RunCache(run_id, run_dir)
            self._caches[run_id] = rc
        return rc

    # -- public queries -----------------------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        """Newest-first run summaries. Cheap: stat + cached manifest + tail probe."""
        result = []
        with self._lock:
            for run_id, run_dir in self._list_run_dirs():
                rc = self._cache_for(run_id, run_dir)
                manifest = rc.manifest()
                events_path = rc.events_path
                try:
                    size = os.path.getsize(events_path)
                except OSError:
                    size = 0
                # Prefer an already-parsed last_ts; otherwise probe the tail.
                last_ts = rc.last_ts if rc.last_ts > 0 else _tail_last_ts(events_path, size)
                result.append(
                    {
                        "id": run_id,
                        "manifest": manifest,
                        "lastEventTs": last_ts,
                        "sizeBytes": size,
                    }
                )
        result.sort(key=lambda r: (r["lastEventTs"], r["id"]), reverse=True)
        return result

    def _refreshed_cache(self, run_id: str) -> Optional[_RunCache]:
        # run_id comes from the URL: refuse anything that could escape the runs root.
        if not run_id or run_id in (".", "..") or any(c in run_id for c in "/\\") or ".." in run_id:
            return None
        run_dir = os.path.join(self.root, run_id)
        if not os.path.isdir(run_dir):
            return None
        rc = self._cache_for(run_id, run_dir)
        rc.refresh()
        return rc

    def meta(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            rc = self._refreshed_cache(run_id)
            if rc is None:
                return None
            return {
                "metrics": sorted(rc.metrics),
                "tagKeys": {name: sorted(keys) for name, keys in rc.tag_keys.items()},
                "maxStep": rc.max_step,
            }

    def series(
        self,
        run_id: str,
        name: str,
        group_by: Optional[str] = None,
        bucket: Any = "auto",
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            rc = self._refreshed_cache(run_id)
            if rc is None:
                return None
            # Copy nothing: bucket_series only reads.
            groups = bucket_series(rc.events, name, group_by=group_by, bucket=bucket)
        return {"groups": groups}
