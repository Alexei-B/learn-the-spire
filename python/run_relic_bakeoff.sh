#!/usr/bin/env bash
# Relic solo bake-off (roadmap M3.5). Each variant is a fresh solo relic run, ~6k steps, same seed/data.
set -u
PY=.venv/Scripts/python.exe
OUT=checkpoints/relic_bakeoff
mkdir -p "$OUT"
COMMON="--arch factored --train-experts relics --val-experts trained-only --steps 6000 \
 --batch 512 --val-every 1000 --val-batch 512 --val-states 3000 --log-every 500 --warmup 800 \
 --seed 0 --cache data/corpus_tok_v3 --corpus data/corpus2 --device cuda --amp bf16"

run () {
  local label="$1"; shift
  echo "======== $label ========"
  $PY -m lts2_agent.train_encdec $COMMON --ckpt "$OUT/$label.pt" --run-label "$label" "$@" \
      > "$OUT/$label.log" 2>&1
  echo "[$label] final VAL:"
  grep "VAL\[" "$OUT/$label.log" | tail -1
  grep "sps=" "$OUT/$label.log" | tail -1
}

run relic_a_pw1   --fac-relic-head set --relic-pos-weight 1
run relic_a_pw5   --fac-relic-head set --relic-pos-weight 5
run relic_a_pw15  --fac-relic-head set --relic-pos-weight 15
run relic_b_deep  --fac-relic-head set --relic-pos-weight 5 --relic-dec-layers 3
run relic_c_lr1e3 --fac-relic-head set --relic-pos-weight 5 --lr 1e-3
run relic_d_slots --fac-relic-head slots

echo "======== BAKE-OFF DONE ========"
