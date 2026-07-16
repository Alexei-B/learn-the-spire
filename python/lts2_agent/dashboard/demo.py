"""Synthetic run generator for the dashboard: ``python -m lts2_agent.dashboard.demo``.

Writes a plausible run directory (manifest.json + events.jsonl) matching the data
contract so the dashboard has something to show without a real training run. With
``--live`` it keeps appending, one iteration every ``--interval`` seconds, so the
live-tailing and "live dot" can be watched end to end.

This is dev/test scaffolding, not part of the trainer — it never imports training
code and only writes the file contract the dashboard reads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, IO

ACTS = ["1", "2", "3"]
CHARACTERS = ["Ironclad", "Necrobinder"]
# Opaque room-tag values spanning monster/elite/boss variants (grouped dynamically).
ROOMS = [
    "monster.cultist",
    "monster.jawworm",
    "monster.louses",
    "elite.gremlin_nob",
    "elite.lagavulin",
    "boss.guardian",
    "boss.hexaghost",
]
EVAL_MODES = ["greedy", "sampled"]


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write(fh: IO[str], ev: dict[str, Any]) -> None:
    fh.write(json.dumps(ev, separators=(",", ":")))
    fh.write("\n")
    fh.flush()


def _room_difficulty(room: str) -> float:
    if room.startswith("boss"):
        return 0.55
    if room.startswith("elite"):
        return 0.72
    return 0.9


def _emit_iteration(fh: IO[str], step: int, rng: random.Random) -> None:
    """Emit one iteration's worth of train.* scalars, per-fight events, evals."""
    ts = time.time()
    # A learning curve that improves then plateaus, plus noise.
    progress = 1.0 - math.exp(-step / 60.0)

    def _s(name: str, value: float) -> None:
        _write(fh, {"ts": ts, "phase": "train", "step": step, "name": name, "value": value})

    _s("train.win_rate", min(0.95, 0.15 + 0.7 * progress + rng.uniform(-0.05, 0.05)))
    _s("train.hp_lost", max(0.0, 40.0 * (1.0 - progress) + rng.uniform(-4, 4)))
    _s("train.loss", max(0.01, 2.0 * math.exp(-step / 45.0) + rng.uniform(-0.05, 0.05)))
    _s("train.entropy", max(0.05, 1.4 * math.exp(-step / 120.0) + rng.uniform(-0.03, 0.03)))
    _s("train.sps", 800.0 + rng.uniform(-60, 60))

    # A handful of per-fight outcome events, tagged act/room/character/truncated.
    for _ in range(rng.randint(4, 8)):
        act = rng.choice(ACTS)
        room = rng.choice(ROOMS)
        character = rng.choice(CHARACTERS)
        p_win = min(0.98, _room_difficulty(room) * (0.4 + 0.6 * progress))
        won = 1.0 if rng.random() < p_win else 0.0
        hp_lost = max(0.0, (1.0 - p_win) * 35.0 + rng.uniform(-6, 8))
        truncated = "true" if rng.random() < 0.05 else "false"
        tags = {"act": act, "room": room, "character": character, "truncated": truncated}
        _write(fh, {"ts": ts, "phase": "fight", "step": step, "name": "fight.won",
                    "value": won, "tags": tags})
        _write(fh, {"ts": ts, "phase": "fight", "step": step, "name": "fight.hp_lost",
                    "value": hp_lost, "tags": tags})

    # Periodic fixed-seed eval: aggregates + per-fight breakdowns.
    if step % 10 == 0:
        for mode in EVAL_MODES:
            bump = 0.05 if mode == "greedy" else 0.0  # greedy usually a touch ahead
            wr = min(0.95, 0.12 + 0.7 * progress + bump + rng.uniform(-0.03, 0.03))
            _write(fh, {"ts": ts, "phase": "eval", "step": step,
                        "name": f"eval.{mode}.win", "value": wr})
            _write(fh, {"ts": ts, "phase": "eval", "step": step,
                        "name": f"eval.{mode}.hp_lost",
                        "value": max(0.0, 40.0 * (1.0 - progress) + rng.uniform(-3, 3))})
            for _ in range(rng.randint(3, 6)):
                act = rng.choice(ACTS)
                room = rng.choice(ROOMS)
                character = rng.choice(CHARACTERS)
                p_win = min(0.98, _room_difficulty(room) * (0.4 + 0.6 * progress) + bump)
                won = 1.0 if rng.random() < p_win else 0.0
                tags = {"act": act, "room": room, "character": character, "mode": mode}
                _write(fh, {"ts": ts, "phase": "eval", "step": step, "name": "eval_fight.won",
                            "value": won, "tags": tags})
                _write(fh, {"ts": ts, "phase": "eval", "step": step, "name": "eval_fight.hp_lost",
                            "value": max(0.0, (1.0 - p_win) * 32.0 + rng.uniform(-5, 6)),
                            "tags": tags})


def _write_manifest(run_dir: str, run_id: str, label: str, seed: int) -> None:
    manifest = {
        "runId": run_id,
        "label": label,
        "startedAt": _iso_now(),
        "kind": "scenario",
        "argv": ["python", "-m", "lts2_agent.dashboard.demo", "--dir", run_dir],
        "config": {"envs": 8, "steps": 96, "mode": "scenario", "seed": seed},
        "gitSha": "demodemo",
        "featureVersion": 7,
        "catalogSignature": "demo-catalog-0001",
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m lts2_agent.dashboard.demo",
        description="Generate a synthetic training run for the dashboard.",
    )
    parser.add_argument("--dir", default="checkpoints/runs", help="Runs root directory.")
    parser.add_argument("--run-id", default=None, help="Run subdir name (default: demo-<ts>).")
    parser.add_argument("--label", default="demo run", help="Human label in the manifest.")
    parser.add_argument("--iterations", type=int, default=200, help="Iterations to write.")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed (deterministic data).")
    parser.add_argument("--live", action="store_true",
                        help="Keep appending forever, one iteration per --interval.")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between iterations in --live mode (default: 2).")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    run_id = args.run_id or f"demo-{int(time.time())}"
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    _write_manifest(run_dir, run_id, args.label, args.seed)
    rng = random.Random(args.seed)
    events_path = os.path.join(run_dir, "events.jsonl")

    print(f"Writing synthetic run to {run_dir}", flush=True)
    with open(events_path, "a", encoding="utf-8") as fh:
        for step in range(args.iterations):
            _emit_iteration(fh, step, rng)
        if args.live:
            step = args.iterations
            print("Live mode: appending an iteration every "
                  f"{args.interval}s (Ctrl+C to stop).", flush=True)
            try:
                while True:
                    time.sleep(args.interval)
                    _emit_iteration(fh, step, rng)
                    step += 1
            except KeyboardInterrupt:
                pass
    print(f"Done: {events_path}", flush=True)


if __name__ == "__main__":
    main()
