#!/usr/bin/env bash
# Well-posedness canary probes (tokenizer v4 / roadmap M3.10). Confirm the fix lifts the ill-posed floors:
#   * potions synth  — expect `expert_exact` (coverage val) to jump WELL above the ~0.46 v3 plateau
#   * orbs   synth   — expect improvement over its old plateau (position now visible)
#   * potions real   — expect the 0.995 floor class (the ~3 rare non-left-packed fights) GONE
# Batch 512 (coexists with the orchestrator sweep at batch<=512); val-every 50 for a fine canary curve.
# Sequential so the three do not over-subscribe the GPU next to the running sweep.
set -u
PY=.venv/Scripts/python.exe
CK=/c/Users/Alexe/AppData/Local/Temp/lts2_wp/ck
LOG=/c/Users/Alexe/AppData/Local/Temp/lts2_wp/logs
mkdir -p "$CK" "$LOG"
COMMON="--arch factored --cache data/corpus_tok_v31 --corpus data/corpus2 --lr 3e-4 --beta2 0.95 \
  --loss-balance expert --val-experts trained-only --num-targets twohot --fac-relic-head slots \
  --batch 512 --steps 6000 --warmup 800 --val-every 50 --val-states 2000 --val-batch 512 --seed 0"

run() {  # label expert data extra...
  local label=$1 expert=$2 data=$3; shift 3
  echo "=== [$(date +%H:%M:%S)] START $label (expert=$expert data=$data $*)"
  $PY -u -m lts2_agent.train_encdec $COMMON \
     --train-experts "$expert" --data "$data" "$@" \
     --ckpt "$CK/${label}.pt" --run-label "$label" > "$LOG/${label}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE  $label (exit $?)"
}

run wp-potions-synth  potions synth
run wp-orbs-synth     orbs    synth
run wp-potions-real   potions real   --focus-present 0.9
echo "=== ALL CANARY PROBES COMPLETE [$(date +%H:%M:%S)]"
