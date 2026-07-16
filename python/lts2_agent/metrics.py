"""Structured per-run metrics event stream (JSONL) for training runs.

A training run writes one directory under ``<run-dir>/<run_id>/`` holding:

- ``manifest.json`` — written once at open: run identity, full CLI/config, git SHA, and the
  feature/catalog version stamps (the parity contract).
- ``events.jsonl`` — append-only, one JSON object per line, flushed per write so a live tail
  (the dashboard) sees events as they happen. Every line is independently parseable, so a
  crash mid-write loses at most the final partial line.

The dashboard is a *separate* process that only reads these files (list runs = list dirs; live =
tail ``events.jsonl``). Nothing here knows the dashboard exists. Stdlib only — the core
``lts2_agent`` modules stay dependency-free.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def _git_sha() -> Optional[str]:
    """Best-effort ``git rev-parse HEAD``; ``None`` if git is unavailable or this isn't a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            return sha or None
    except Exception:
        pass
    return None


def _json_safe(value: Any) -> Any:
    """Coerce a config value into something ``json.dumps`` accepts, lossily if need be."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def make_run_id(label: str, now: Optional[datetime] = None) -> str:
    """``<UTC yyyymmdd-HHMMSS>-<label>`` — sortable timestamp prefix + human label."""
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{label}"


class MetricsWriter:
    """Writes ``manifest.json`` once and appends events to ``events.jsonl`` (flushed per write).

    Usage::

        with MetricsWriter(run_dir="checkpoints/runs", label="necro", argv=sys.argv,
                           config=vars(args)) as mw:
            mw.emit("train", it, "train.win_rate", 0.42)
            mw.emit("train", it, "fight.won", 1.0, tags={"act": "1", "room": "Monster"})

    ``MetricsWriter(..., enabled=False)`` is a no-op sink (for ``--no-metrics``) so callers never
    branch on whether metrics are on.
    """

    def __init__(self, run_dir: str, label: str, argv: list[str], config: Mapping[str, Any],
                 *, kind: str = "ppo-scenario", feature_version: Any = None,
                 catalog_signature: Any = None, enabled: bool = True,
                 started_at: Optional[datetime] = None):
        self.enabled = enabled
        self._fh = None
        if not enabled:
            self.run_id = None
            self.run_dir = None
            return

        started_at = started_at or datetime.now(timezone.utc)
        self.run_id = make_run_id(label, started_at)
        self.run_dir = os.path.join(run_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        manifest = {
            "runId": self.run_id,
            "label": label,
            "startedAt": started_at.isoformat(),
            "kind": kind,
            "argv": list(argv),
            "config": {str(k): _json_safe(v) for k, v in config.items()},
            "gitSha": _git_sha(),
            "featureVersion": _json_safe(feature_version),
            "catalogSignature": _json_safe(catalog_signature),
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        self._fh = open(os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8")

    def emit(self, phase: str, step: int, name: str, value: float,
             tags: Optional[Mapping[str, str]] = None) -> None:
        """Append one event line and flush it. No-op when disabled."""
        if not self.enabled or self._fh is None:
            return
        event: dict[str, Any] = {
            "ts": time.time(),
            "phase": phase,
            "step": int(step),
            "name": name,
            "value": float(value),
        }
        if tags:
            event["tags"] = {str(k): str(v) for k, v in tags.items()}
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
