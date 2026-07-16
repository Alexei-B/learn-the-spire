"""MetricsWriter contract tests: manifest schema, per-line JSONL events, and live (per-write) flush.

No C# environment needed — this validates the file contract the (separate) dashboard reads.
"""

from __future__ import annotations

import json
import os

from lts2_agent.metrics import MetricsWriter, make_run_id


def _read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_id_shape():
    rid = make_run_id("necro")
    stamp, _, label = rid.partition("-")[0], "-", rid
    assert label.endswith("-necro")
    # <yyyymmdd>-<HHMMSS>-<label>
    parts = rid.split("-")
    assert len(parts[0]) == 8 and parts[0].isdigit()
    assert len(parts[1]) == 6 and parts[1].isdigit()


def test_manifest_written_once(tmp_path):
    args = {"envs": 4, "device": "cpu", "lr": 3e-4, "flag": True, "obj": object()}
    with MetricsWriter(run_dir=str(tmp_path), label="lbl", argv=["a", "b"], config=args,
                       feature_version=5, catalog_signature="sig123") as mw:
        run_dir = mw.run_dir
        assert os.path.isdir(run_dir)
        with open(os.path.join(run_dir, "manifest.json"), "r", encoding="utf-8") as f:
            man = json.load(f)

    assert man["runId"] == os.path.basename(run_dir)
    assert man["label"] == "lbl"
    assert man["kind"] == "ppo-scenario"
    assert man["argv"] == ["a", "b"]
    assert man["featureVersion"] == 5
    assert man["catalogSignature"] == "sig123"
    assert "gitSha" in man  # best-effort: str or None
    assert man["startedAt"].endswith("+00:00")  # ISO 8601 UTC
    # config is JSON-safe: the non-serializable object was coerced to str.
    assert man["config"]["envs"] == 4 and man["config"]["flag"] is True
    assert isinstance(man["config"]["obj"], str)


def test_events_are_line_parseable_with_schema(tmp_path):
    with MetricsWriter(run_dir=str(tmp_path), label="lbl", argv=[], config={}) as mw:
        events_path = os.path.join(mw.run_dir, "events.jsonl")
        mw.emit("train", 1, "train.win_rate", 0.5)
        mw.emit("train", 1, "fight.won", 1.0, tags={"act": "1", "room": "Monster", "character": "IRONCLAD"})
        mw.emit("eval", 2, "eval.greedy.win", 0.25, tags={})  # empty tags -> omitted

    events = _read_events(events_path)
    assert len(events) == 3
    for e in events:
        assert set(["ts", "phase", "step", "name", "value"]).issubset(e)
        assert isinstance(e["ts"], float)
        assert isinstance(e["step"], int)
        assert isinstance(e["value"], float)
    assert events[0] == {"ts": events[0]["ts"], "phase": "train", "step": 1,
                         "name": "train.win_rate", "value": 0.5}
    assert events[1]["tags"] == {"act": "1", "room": "Monster", "character": "IRONCLAD"}
    assert "tags" not in events[2]  # empty tags omitted


def test_flush_is_per_write_readable_while_open(tmp_path):
    """The dashboard tails events.jsonl live, so each emit must be flushed and independently
    parseable *before* close()."""
    with MetricsWriter(run_dir=str(tmp_path), label="lbl", argv=[], config={}) as mw:
        events_path = os.path.join(mw.run_dir, "events.jsonl")
        mw.emit("train", 1, "a", 1.0)
        assert len(_read_events(events_path)) == 1  # readable while still open
        mw.emit("train", 2, "b", 2.0)
        assert len(_read_events(events_path)) == 2


def test_disabled_writer_is_noop(tmp_path):
    with MetricsWriter(run_dir=str(tmp_path), label="lbl", argv=[], config={}, enabled=False) as mw:
        assert mw.run_dir is None
        mw.emit("train", 1, "a", 1.0)  # must not raise
    # nothing written
    assert not os.listdir(tmp_path)
