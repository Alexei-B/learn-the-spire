"""Dashboard server logic: bucketing, group-by, incremental tail, truncation tolerance.

No C# environment and no third-party deps — the dashboard is stdlib-only, so these
tests exercise the store directly plus the HTTP API over a real ephemeral socket.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from lts2_agent.dashboard import bucket_series
from lts2_agent.dashboard.server import make_server
from lts2_agent.dashboard.store import RunStore


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ev(name, step, value, tags=None, ts=1.0):
    ev = {"ts": ts, "phase": "train", "step": step, "name": name, "value": value}
    if tags:
        ev["tags"] = tags
    return json.dumps(ev, separators=(",", ":")) + "\n"


def _write_run(root, run_id, lines, manifest=None):
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    if manifest is None:
        manifest = {"runId": run_id, "label": run_id, "startedAt": "2026-07-16T00:00:00Z",
                    "kind": "scenario"}
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    return run_dir


# --------------------------------------------------------------------------
# bucketing
# --------------------------------------------------------------------------
def test_bucket_means_and_counts():
    events = [
        {"name": "m", "step": 0, "value": 0.0, "tags": {}},
        {"name": "m", "step": 1, "value": 1.0, "tags": {}},
        {"name": "m", "step": 2, "value": 2.0, "tags": {}},
        {"name": "m", "step": 3, "value": 3.0, "tags": {}},
    ]
    groups = bucket_series(events, "m", group_by="none", bucket=2)
    pts = groups["all"]
    assert len(pts) == 2
    # bucket 0: steps 0,1 -> mean 0.5, n 2 ; bucket 1: steps 2,3 -> mean 2.5, n 2
    assert pts[0]["value"] == 0.5 and pts[0]["n"] == 2
    assert pts[1]["value"] == 2.5 and pts[1]["n"] == 2
    # n is present on every point
    assert all("n" in p for p in pts)


def test_bucket_filters_by_name():
    events = [
        {"name": "a", "step": 0, "value": 5.0, "tags": {}},
        {"name": "b", "step": 0, "value": 99.0, "tags": {}},
    ]
    groups = bucket_series(events, "a", group_by="none", bucket=1)
    assert groups == {"all": [{"step": 0, "value": 5.0, "n": 1}]}


def test_bucket_auto_targets_leq_500_points():
    events = [{"name": "m", "step": s, "value": float(s), "tags": {}} for s in range(5000)]
    groups = bucket_series(events, "m", group_by="none", bucket="auto")
    assert len(groups["all"]) <= 500
    # total sample count preserved
    assert sum(p["n"] for p in groups["all"]) == 5000


def test_group_by_tag_value():
    events = [
        {"name": "fight.won", "step": 0, "value": 1.0, "tags": {"room": "boss"}},
        {"name": "fight.won", "step": 0, "value": 0.0, "tags": {"room": "boss"}},
        {"name": "fight.won", "step": 0, "value": 1.0, "tags": {"room": "monster"}},
    ]
    groups = bucket_series(events, "fight.won", group_by="room", bucket=1)
    assert set(groups.keys()) == {"boss", "monster"}
    assert groups["boss"][0]["value"] == 0.5 and groups["boss"][0]["n"] == 2
    assert groups["monster"][0]["value"] == 1.0 and groups["monster"][0]["n"] == 1


def test_group_by_missing_tag_falls_in_untagged():
    events = [
        {"name": "m", "step": 0, "value": 1.0, "tags": {"act": "1"}},
        {"name": "m", "step": 0, "value": 2.0, "tags": {}},
    ]
    groups = bucket_series(events, "m", group_by="act", bucket=1)
    assert groups["1"][0]["value"] == 1.0
    assert groups["(untagged)"][0]["value"] == 2.0


# --------------------------------------------------------------------------
# incremental tail + truncation tolerance
# --------------------------------------------------------------------------
def test_incremental_tail_reads_only_appended(tmp_path):
    root = str(tmp_path)
    _write_run(root, "r1", [_ev("train.loss", 0, 1.0)])
    store = RunStore(root)

    meta = store.meta("r1")
    assert meta["maxStep"] == 0
    assert meta["metrics"] == ["train.loss"]

    # append more, then refresh via another query
    with open(os.path.join(root, "r1", "events.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(_ev("train.loss", 5, 0.5))
        fh.write(_ev("train.win_rate", 5, 0.9))
    meta = store.meta("r1")
    assert meta["maxStep"] == 5
    assert set(meta["metrics"]) == {"train.loss", "train.win_rate"}

    ser = store.series("r1", "train.loss", group_by="none", bucket=1)
    steps = [p["step"] for p in ser["groups"]["all"]]
    assert steps == [0, 5]


def test_truncated_final_line_tolerated_then_completed(tmp_path):
    root = str(tmp_path)
    path_dir = os.path.join(root, "r1")
    os.makedirs(path_dir)
    with open(os.path.join(path_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"runId": "r1", "label": "r1"}, fh)
    events = os.path.join(path_dir, "events.jsonl")

    good = _ev("m", 0, 1.0)
    # a full good line, then a mid-write fragment with no trailing newline
    fragment = '{"ts":1.0,"phase":"train","step":1,"name":"m","val'
    with open(events, "w", encoding="utf-8") as fh:
        fh.write(good + fragment)

    store = RunStore(root)
    ser = store.series("r1", "m", group_by="none", bucket=1)
    # only the complete line is counted; the fragment is held back
    assert [p["step"] for p in ser["groups"]["all"]] == [0]

    # now the writer finishes that line and adds another
    with open(events, "a", encoding="utf-8") as fh:
        fh.write('ue":2.0}\n')
        fh.write(_ev("m", 2, 3.0))
    ser = store.series("r1", "m", group_by="none", bucket=1)
    assert [p["step"] for p in ser["groups"]["all"]] == [0, 1, 2]


def test_corrupt_complete_line_skipped(tmp_path):
    root = str(tmp_path)
    _write_run(root, "r1", [_ev("m", 0, 1.0), "not json at all\n", _ev("m", 1, 2.0)])
    store = RunStore(root)
    ser = store.series("r1", "m", group_by="none", bucket=1)
    assert [p["step"] for p in ser["groups"]["all"]] == [0, 1]


def test_file_shrink_resets(tmp_path):
    root = str(tmp_path)
    events = _write_run(root, "r1", [_ev("m", s, float(s)) for s in range(5)])
    events_path = os.path.join(events, "events.jsonl")
    store = RunStore(root)
    assert store.meta("r1")["maxStep"] == 4
    # rotate: overwrite with a shorter file
    with open(events_path, "w", encoding="utf-8") as fh:
        fh.write(_ev("m", 0, 9.0))
    meta = store.meta("r1")
    assert meta["maxStep"] == 0
    ser = store.series("r1", "m", group_by="none", bucket=1)
    assert ser["groups"]["all"] == [{"step": 0, "value": 9.0, "n": 1}]


# --------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------
def test_list_runs_newest_first(tmp_path):
    root = str(tmp_path)
    _write_run(root, "old", [_ev("m", 0, 1.0, ts=100.0)])
    _write_run(root, "new", [_ev("m", 0, 1.0, ts=200.0)])
    store = RunStore(root)
    runs = store.list_runs()
    ids = [r["id"] for r in runs]
    assert ids == ["new", "old"]
    assert runs[0]["lastEventTs"] == 200.0
    assert runs[0]["sizeBytes"] > 0
    assert runs[0]["manifest"]["runId"] == "new"


def test_run_appearing_between_polls(tmp_path):
    root = str(tmp_path)
    store = RunStore(root)
    assert store.list_runs() == []
    _write_run(root, "late", [_ev("m", 0, 1.0)])
    runs = store.list_runs()
    assert [r["id"] for r in runs] == ["late"]


def test_meta_missing_run(tmp_path):
    store = RunStore(str(tmp_path))
    assert store.meta("nope") is None
    assert store.series("nope", "m") is None


# --------------------------------------------------------------------------
# HTTP API over a real socket
# --------------------------------------------------------------------------
def _serve(root):
    httpd = make_server(root, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    return httpd, "http://127.0.0.1:%d" % port


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_http_endpoints(tmp_path):
    root = str(tmp_path)
    _write_run(root, "r1", [
        _ev("train.loss", 0, 1.0),
        _ev("fight.won", 0, 1.0, tags={"room": "boss", "act": "1"}),
        _ev("fight.won", 0, 0.0, tags={"room": "boss", "act": "1"}),
        _ev("fight.won", 0, 1.0, tags={"room": "monster", "act": "2"}),
    ])
    httpd, base = _serve(root)
    try:
        status, runs = _get(base, "/api/runs")
        assert status == 200 and runs[0]["id"] == "r1"

        status, meta = _get(base, "/api/runs/r1/meta")
        assert status == 200
        assert "fight.won" in meta["metrics"]
        assert set(meta["tagKeys"]["fight.won"]) == {"room", "act"}

        status, ser = _get(base, "/api/runs/r1/series?name=fight.won&group_by=room&bucket=1")
        assert status == 200
        assert ser["groups"]["boss"][0]["value"] == 0.5
        assert ser["groups"]["boss"][0]["n"] == 2
        assert ser["groups"]["monster"][0]["value"] == 1.0

        # index served at /
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert "Lts2 training" in body

        # missing metric -> 400
        try:
            _get(base, "/api/runs/r1/series")
            assert False, "expected 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_live_append_visible(tmp_path):
    root = str(tmp_path)
    events_dir = _write_run(root, "r1", [_ev("m", 0, 1.0)])
    httpd, base = _serve(root)
    try:
        _, meta = _get(base, "/api/runs/r1/meta")
        assert meta["maxStep"] == 0
        with open(os.path.join(events_dir, "events.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_ev("m", 42, 2.0))
        _, meta = _get(base, "/api/runs/r1/meta")
        assert meta["maxStep"] == 42
    finally:
        httpd.shutdown()
        httpd.server_close()
