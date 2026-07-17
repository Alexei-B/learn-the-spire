"""Parse the synthetic-space probe run metrics into a report table.

For each run label, read checkpoints/runs/<newest matching>/events.jsonl and pull the final (and a matched
step) eval.expert_dist / eval.expert_exact for both the REAL val (untagged) and the synthetic COVERAGE val
(tag val=coverage), plus relic_set_f1. Prints a markdown table for the roadmap/report.
"""
import glob
import json
import os
import sys

RUNS = "checkpoints/runs"


def newest_run(label):
    cand = [d for d in glob.glob(f"{RUNS}/*-{label}") if os.path.isdir(d)]
    if not cand:
        return None
    return max(cand, key=os.path.getmtime)


def series(run_dir, name, expert, val=None):
    """List of (step, value) for eval.<name> filtered by expert tag and optional val tag."""
    out = []
    p = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        e = json.loads(line)
        if e.get("phase") != "eval" or e.get("name") != f"eval.{name}":
            continue
        tags = e.get("tags") or {}
        if expert is not None and tags.get("expert") != expert:
            continue
        if val is None and "val" in tags:
            continue
        if val is not None and tags.get("val") != val:
            continue
        out.append((e["step"], e["value"]))
    return out


def final(run_dir, name, expert, val=None):
    s = series(run_dir, name, expert, val)
    return s[-1] if s else (None, None)


def main():
    # label -> (expert, data-desc)
    probes = [
        ("synth-potions-real", "potions", "real"),
        ("synth-potions-synth", "potions", "synth"),
        ("synth-potions-mixed", "potions", "mixed:0.5"),
        ("synth-relics-real", "relics", "real"),
        ("synth-relics-synth", "relics", "synth"),
        ("synth-relics-mixed", "relics", "mixed:0.5"),
        ("synth-orbs-real", "orbs", "real"),
        ("synth-orbs-synth", "orbs", "synth"),
        ("synth-orbs-mixed", "orbs", "mixed:0.5"),
        ("synth-cards-real", "cards", "real"),
        ("synth-cards-mixed25", "cards", "mixed:0.25"),
        ("synth-cards-mixed50", "cards", "mixed:0.5"),
    ]
    print("| expert | data | step | real-val exact ↑ | real-val dist ↓ | coverage exact ↑ | coverage dist ↓ | relic_f1 (real/cov) |")
    print("|---|---|---|---|---|---|---|---|")
    for label, expert, data in probes:
        rd = newest_run(label)
        if rd is None:
            print(f"| {expert} | {data} | — | (no run) | | | | |")
            continue
        st, rex = final(rd, "expert_exact", expert, None)
        _, rdist = final(rd, "expert_dist", expert, None)
        _, cex = final(rd, "expert_exact", expert, "coverage")
        _, cdist = final(rd, "expert_dist", expert, "coverage")
        rf1 = final(rd, "relic_set_f1", None, None)[1]
        cf1 = final(rd, "relic_set_f1", None, "coverage")[1]

        def f(x):
            return f"{x:.4f}" if isinstance(x, float) else "—"
        f1s = f"{f(rf1)}/{f(cf1)}" if expert == "relics" else "—"
        print(f"| {expert} | {data} | {st or '—'} | {f(rex)} | {f(rdist)} | {f(cex)} | {f(cdist)} | {f1s} |")


if __name__ == "__main__":
    main()
