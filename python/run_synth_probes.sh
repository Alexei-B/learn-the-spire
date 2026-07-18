#!/usr/bin/env bash
# Sequential synthetic-space training probes (roadmap M3.5). Batch 512 to coexist with the orchestrator's
# batch-1536 creatures runs. Each run writes metrics (events.jsonl) under checkpoints/runs/<label> and a
# stdout log under the temp logs dir. Ordered so the KEY-question runs (potions floor, relics plateau)
# finish first.
set -u
PY=.venv/Scripts/python.exe
CK=/c/Users/Alexe/AppData/Local/Temp/lts2_synth/ck
LOG=/c/Users/Alexe/AppData/Local/Temp/lts2_synth/logs
COMMON="--arch factored --corpus data/corpus2 --lr 3e-4 --beta2 0.95 \
  --loss-balance expert --val-experts trained-only --num-targets twohot \
  --batch 512 --steps 6000 --warmup 800 --val-every 300 --val-states 2000 --val-batch 512 --seed 0"

run() {  # label expert data extra...
  local label=$1 expert=$2 data=$3; shift 3
  echo "=== [$(date +%H:%M:%S)] START $label (expert=$expert data=$data $*)"
  $PY -u -m lts2_agent.train_encdec $COMMON \
     --train-experts "$expert" --data "$data" "$@" \
     --ckpt "$CK/${label}.pt" --run-label "$label" > "$LOG/${label}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE  $label (exit $?)"
}

# expert  |  real (focus 0.9)      synth                 mixed
run synth-potions-synth   potions synth
run synth-potions-real    potions real   --focus-present 0.9
run synth-relics-synth    relics  synth
run synth-relics-real     relics  real   --focus-present 0.9
run synth-orbs-synth      orbs    synth
run synth-orbs-real       orbs    real   --focus-present 0.9
run synth-potions-mixed   potions mixed:0.5
run synth-relics-mixed    relics  mixed:0.5
run synth-orbs-mixed      orbs    mixed:0.5
run synth-cards-mixed50   cards   mixed:0.5
run synth-cards-mixed25   cards   mixed:0.25
run synth-cards-real      cards   real   --focus-present 0.9
echo "=== ALL PROBES COMPLETE [$(date +%H:%M:%S)]"
