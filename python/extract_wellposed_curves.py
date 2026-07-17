"""Extract the well-posedness canary probe curves (tokenizer v4) from the wp-* run event logs.

For each canary run, print eval.expert_exact / eval.expert_dist over steps for BOTH the real fixed val
(untagged) and the synthetic COVERAGE val (tag val=coverage) — the coverage-jump verdict vs the v3 floor.
"""
import glob
import json
import os

RUNS = "checkpoints/runs"
LABELS = [("wp-potions-synth", "potions"), ("wp-orbs-synth", "orbs"), ("wp-potions-real", "potions")]


def newest_run(label):
    cand = [d for d in glob.glob(f"{RUNS}/*-{label}") if os.path.isdir(d)]
    return max(cand, key=os.path.getmtime) if cand else None


def series(run_dir, name, expert, val=None):
    out = []
    p = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            e = json.loads(line)
        except Exception:
            continue
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


def fmt(s):
    if not s:
        return "(none)"
    # sample a handful of steps across the run
    pts = s if len(s) <= 8 else [s[0]] + [s[i] for i in (len(s)//4, len(s)//2, 3*len(s)//4)] + [s[-1]]
    return "  ".join(f"{st}:{v:.4f}" for st, v in pts)


for label, expert in LABELS:
    rd = newest_run(label)
    print("=" * 100)
    print(f"{label}  (expert={expert})  -> {rd}")
    if rd is None:
        print("  (no run dir)")
        continue
    for val, tag in ((None, "REAL val    "), ("coverage", "COVERAGE val")):
        ex = series(rd, "expert_exact", expert, val)
        di = series(rd, "expert_dist", expert, val)
        print(f"  {tag} expert_exact: {fmt(ex)}")
        print(f"  {tag} expert_dist : {fmt(di)}")
