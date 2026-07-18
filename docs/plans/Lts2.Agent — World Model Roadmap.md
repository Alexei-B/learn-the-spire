# Lts2.Agent — World-Model Roadmap (implementation backlog)

Status: **M0–M3 done (CP1–CP3 approved; CP4 package assembled, pending review).** M3 headline:
the M4 start-gate is PASSED — gate-run best checkpoint (`wm_gate_v2.pt.best`, step 73.5k: tokens
latent + tokenizer-v2 + twohot + lr 6e-4, plain cosine) reconstructs held-out states at
`state_dist` 0.0297 = **action-SNR 5.74** (gate ≥4), card-id 0.992, zone/power-id/energy ≈1.0,
HP MAE 0.96, first nonzero exact reconstructions (mech 0.0013). Legal actions derived from
*decoded* states: exact-set 0.876 / F1 0.972 (true-state bound 0.998/0.999). Residual mismatch:
creatures 35% / cards 35% / **relics 19%** (relic slots decode with duplicates — F1 0.905, the
next structural target). _Relic model corrected (tokenizer v5 / v3.2, 2026-07-18):_ two product facts
overturned the earlier duplicate-free treatments — relic ORDER is semantic (wax relics like Tezcatara
expire in acquisition order) and duplicate relics DO occur (measured 3238/4.0M `data/corpus2` states,
max 2 copies). Relics are therefore now a **POSITIONAL** type (the orb treatment): one token per relic
INSTANCE, wire order preserved (v3.1's `relics.sort()` reverted), an explicit `slot` acquisition-order
column, `MAX_RELICS` raised 24→40 bounding TOTAL relics. The set-membership head, its cardinality head,
`_dedup_slot_ids`/`_decode_set_head`, and the `--relic-head`/`--fac-relic-head`/`--dedup` flags are all
**deleted** (the set-vs-slots bake-off below is now moot — relics ride the standard per-slot decode).
Findings: gate-run-v1 collapsed at step 63k under sustained mid-LR on
the stretched schedule (its step-51k best was lost to in-place checkpointing — best-val `.best`
sidecar now prevents recurrence); LR ladder says 6e-4 is the ceiling (1e-3+ degrade smoothly);
EMA at 0.999 was neutral-to-slightly-worse (run still improving at end). Corpus doubled to 2.0M
transitions (corpus-v1b); combined 4M-state cache building. CP3 verdict below: PPO-on-tokens
overlaps the baseline on the live dashboard comparison — parity confirms the tokenizer carries at
least the hand-features' signal with zero feature engineering (the synergy payoff is expected from
the supervised M3/M4 modules, not from PPO); the comparison run was stopped early at the product
owner's direction. This is the implementation plan for
`docs/design/Lts2.Agent — World Model.md` (read that first; this doc assumes its vocabulary:
tokenizer, encoder/decoder, predictor, afterstate/chance step, value/policy, planner).

Scope: container-level architecture, the contracts between containers, milestone ordering, and the
QA process — **not** class/file-level design. The "how" inside each container is the implementor's
to decide per item, following the repo's usual loop (read docs → grep `refsrc/` for game APIs →
implement one shippable slice → seeded tests → update docs → commit).

Conventions here match the harness roadmap: work top-to-bottom, the next task is the next
unchecked item; flip items to _done_ with a note when they land; keep this doc honest.

---

## Goals (from the product owner, condensed)

1. **Model:** implement the world-model architecture in the design doc's phases; decide the open
   analysis questions (flat vs token latents, codebook shape, …) empirically, not up front.
2. **TUI as agent debugger:** the TUI's job for this effort is *manual analysis of agent
   behaviour* — show the full action ranking (not just the Tab pick) and, once the predictor
   exists, each action's predicted next state; make failure scenarios findable by hand.
3. **Precision training monitoring:** a local web dashboard over live + historical training runs.
   Per-phase metrics (reconstruction accuracy, prediction accuracy, value calibration) alongside
   win %/HP-lost, with **breakdowns by act and by monster/elite/boss**. Must work in real time for
   any training run, including ones Claude launches in the background.
4. **Realistic training distribution:** keep broad-random states for *model* (encoder/predictor)
   training, but train the *decision* components on decks resembling real act-1 play: the
   character's starter deck with 0–3 random removals and 0–3 random additions, additions weighted
   **60% own-character pool / 25% colorless / 12% curses / 3% off-character**, never status cards.
   Explicitly **no** scripted fights or fixed hands in training — probe/closed-eval sets are
   evaluation-only, excluded from all training data. Randomized inputs are the overfitting guard.

## Guiding principles (the ordering logic)

- **Tooling before model.** The dashboard, the metric breakdowns, and the TUI ranking view all
  work against the *existing* PPO stack (the decision protocol already carries per-option
  `score`/`rationale`; the trainer already knows act/room per outcome). Building them first (M0)
  means they are debugged against a known system before the new model needs them — and every
  later milestone lands with its instruments already on.
- **Everything observable.** Rule: a learning component may not start training until its metrics
  are flowing to the dashboard. No more judging runs by stdout.
- **Two data regimes, one pipeline.** Scenario generation gains a `deckSpec` (broad-random |
  realistic | explicit). Model-corpus collection uses mostly broad; RL phases use mostly
  realistic; both flow through the same collector, corpus format, and metrics.
- **Seeded determinism end to end.** Same seed + same spec ⇒ same deck, same fight, same corpus
  shard. Train/val/test splits by seed-hash so leakage is structurally impossible.
- **Manual checkpoints (CP1–CP7).** Each milestone ends with a hands-on review gate: a concrete
  thing to run, look at, and judge before the next milestone starts.

## Container view (C4-ish)

```
┌───────────────────────────── C# (.NET 9) ─────────────────────────────┐
│                                                                       │
│  Lts2.Harness ──────────── Lts2.AgentHost                             │
│  (game logic, GameHost,    (env server: reset/step/reset_combat;      │
│   CombatScenario)          + NEW deckSpec scenario gen;               │
│        │                   + catalog dumps: cards, NEW powers)        │
│        │                          ▲ env protocol (JSONL/stdio)        │
│  Lts2.Tui ── Lts2.Agent ──────────┼───────────────────────────────┐   │
│  (debug views:  (ProcessDecision  │                               │   │
│   NEW ranking    Engine, protocol │                               │   │
│   panel, NEW     v1 → NEW v2)     │                               │   │
│   prediction     │ decision protocol (JSONL/stdio)                │   │
│   inspector)     ▼                │                               │   │
└──────────────────┼────────────────┼───────────────────────────────┼───┘
                   │                │                               │
┌──────────────────┼──── Python ────┼───────────────────────────────┼───┐
│  decision_server (serves policy   │   collectors (broad/realistic │   │
│   + NEW explanations/predictions) │    regimes, mixed policies) ──┘   │
│        ▲                          │        │                          │
│        │                          │        ▼                          │
│  world-model stack: tokenizer → encoder/decoder → predictor →         │
│   value/policy → planner   (trainers per phase, checkpoints)          │
│        │                  ▲                                           │
│        │                  │ reads                                     │
│        │           corpus store (sharded transitions, split by seed)  │
│        │           oracle prober (replay-based ground truth, eval-only│
│        ▼                                                              │
│  metrics events (JSONL per run) ──► dashboard web app (separate       │
│                                     process; run list, live charts,   │
│                                     tag breakdowns, run compare)      │
└───────────────────────────────────────────────────────────────────────┘
```

Container responsibilities:

- **Lts2.AgentHost (C#)** — unchanged role (environment server), extended with: `deckSpec`
  scenario generation (single source of truth for deck construction, seeded), a power catalog
  dump, and whatever card-pool metadata (`character pool / colorless / curse`) the weighted
  sampler needs in `cards.json`.
- **Lts2.Agent + Lts2.Tui (C#)** — decision protocol v2 (backward compatible) and the two debug
  views. The TUI stays a *consumer* of agent output; no model code in C#.
- **Python collectors + corpus store** — turn env rollouts (any policy, any regime) into
  append-only transition shards with scenario metadata; own the train/val/test split discipline.
- **Python world-model stack** — the design doc's modules, one trainer per phase, shared
  checkpoint/versioning conventions (tokenizer version supersedes `FEATURE_VERSION` as the
  parity contract).
- **Metrics store + dashboard** — trainer writes JSONL events; the dashboard is a separate local
  web process that only reads files. That decoupling is what makes "watch a run Claude started"
  work with zero coordination, live or after the fact.
- **Oracle prober** — replay-based ground truth (fixed seeds, replayed prefixes) for predictor
  scoring and planner regret. Evaluation-only by construction.

## Contracts (the API design that must be agreed before the milestones that use it)

**1. Metrics event stream** (M0). One JSONL file per run under `checkpoints/runs/<run_id>/`:
`events.jsonl` — `{ts, phase, step, name, value, tags?}` where `tags` carries at least
`{act, room, character}` on outcome events — plus `manifest.json` (full CLI/config, git SHA,
tokenizer/feature version, start time). The dashboard treats the directory as the database:
list runs = list dirs; live = tail the file. Nothing in the trainer knows the dashboard exists.

**2. Dashboard API** (M0). Local HTTP server: `GET /runs` (manifests), `GET /runs/{id}/series?
name=…&group_by=<tag>` (downsampled series), one static single-page UI (no external CDNs; it must
work offline). Auto-refresh by polling; overlay multiple runs on one chart for comparison.

**3. Scenario `deckSpec`** (M1). A field on `reset_combat`, implemented C#-side:

```json
{"kind": "random",    "cards": 15}                             // today's behavior, explicit
{"kind": "realistic", "removals": [0,3], "additions": [0,3],
 "weights": {"own": 0.60, "colorless": 0.25, "curse": 0.12, "offCharacter": 0.03}}
{"kind": "explicit",  "cards": ["StrikeIronclad", "..."]}      // closed-eval only
```

Status cards are never dealt into decks. All sampling from the fight seed. The observation's
`info` block gains the scenario metadata (deckSpec kind, act, room class) so collectors and
metrics tag outcomes without re-deriving.

**4. Corpus schema** (M1). Sharded compressed JSONL: one record per decision =
`{seed, scenarioMeta, t, state, options, actionTaken, nextState, nextOptions, rewardComponents,
done, info}`. Records are exactly the wire observations (lossless, replayable through the
tokenizer forever). Split assignment = hash(fight seed) → train/val/test.

**5. Decision protocol v2** (M4). Backward compatible: request gains `"explain": true`; reply
entries may add `probability`, `value`, and `prediction` — a *decoded, compact* next-state summary
(per-entity fields, not latents) plus, for stochastic actions, top-k chance outcomes with
probabilities. `protocolVersion: 2`; a v1 agent or a reply without `prediction` degrades to
today's behavior. Payload size is a watch-item (predictions for ~20 options × decoded states).

**6. Versioning.** `TOKENIZER_VERSION` (+ catalog signatures) stamps corpora, checkpoints, and
protocol-v2 explanations; any mismatch rejects loudly, exactly like `FEATURE_VERSION` today.

---

## Milestones

### M0 — Instruments first (works entirely against the current PPO stack)

- [x] **0.1 Metrics events** — _done_: `lts2_agent.metrics.MetricsWriter` (stdlib-only) writes
      `manifest.json` + per-line-flushed `events.jsonl` under `checkpoints/runs/<run_id>/`;
      `train_torch` emits per-iteration `train.*`, per-fight `fight.*` tagged
      act/room/character/truncated, `eval.*`/`eval_fight.*` (mode=greedy|sampled), and
      `bodyguard.pass`. Original item: `train_torch` (and the eval loops) emit the event stream (contract 1)
      alongside the existing stdout/CSV; outcome events tagged act/room/character.
- [x] **0.2 Dashboard MVP** — _done._ `python/lts2_agent/dashboard/` (stdlib-only, offline): a
      ThreadingHTTPServer + one self-contained `index.html` (inline CSS/JS, hand-rolled SVG line
      charts). Run list with live dot, checkbox multi-select overlay/compare, metric/group-by/bucket
      toolbar, 2s polling with pause. Reads the pinned file contract (contract 2) only — incremental
      byte-offset tailing, truncated-final-line tolerant. API: `/api/runs`, `/api/runs/<id>/meta`,
      `/api/runs/<id>/series`. Demo generator `python -m lts2_agent.dashboard.demo [--live]`; unit
      tests in `python/tests/test_dashboard.py`. (Consumes 0.1's event stream, being wired in
      parallel; the two touch only the on-disk contract.)
- [x] **0.3 Breakdown views** — _done._ Group-by any tag key (act / room / character / mode) with
      preset buttons "Win by room", "Win by act", "HP lost by room", "Eval greedy vs sampled win";
      covers training outcomes (`fight.*`) and fixed-seed eval (`eval_fight.*`). Sample count `n` is
      mandatory on every series point and shown in legend + tooltip, so low-count rates read as thin.
- [x] **0.4 TUI ranking panel** — _done._ A toggleable panel (`r` hotkey / **View ▸ Agent Ranking**)
      renders the active Strategy engine's full scored ranking for the current decision: options sorted by
      score, each with its score + rationale, the `Tab` pick marked ▸, and explicit **declined** /
      **evaluating** / **no strategy** text; the engine name is in the panel title. Works with the built-in
      `RulesDecisionEngine` and any external `ProcessDecisionEngine` (PPO checkpoint). The ranking is
      fetched once per decision point off the UI thread and shared by both `Tab` and the panel — no extra
      round-trip per keystroke, never blocking the UI; a dead/timed-out agent degrades to "declined".
      Pure formatter `RankingPanel` covered by `RankingPanelTests`.
- [x] **0.5 Baseline capture** — _done_: run `20260716-093447-baseline-ppo` (300 iters, random
      character, acts 0–2, eval every 10, ckpt `checkpoints/baseline_m0.pt`): 45.7k events, final
      fixed-seed eval greedy win 0.38 / sampled 0.75, the M5 comparison bar. Original item: one
      PPO training run + fixed-seed eval recorded through the new
      pipeline, kept as the comparison baseline for M5/M6.

**CP1 (manual review):** start a PPO training run; open the dashboard; watch it live; inspect the
act/room breakdowns; drive a TUI fight with the ranking panel against the PPO agent. Judge: is
this the debugging experience you wanted? Anything missing gets fixed *now*, while iteration is
cheap and the system under observation is well-understood.

### M1 — Data foundation: scenario generator + corpus (design P0)

- [x] **1.1 `deckSpec` scenario generation** (contract 3) in C# — _done._ `CombatScenario.DeckSpec`
      (`Random`/`Realistic`/`Explicit`) is the single seeded source of truth for deck construction, driven
      by an optional `deckSpec` field on `reset_combat` (parsed in `TrainingEnvironmentServer`); absent =
      the prior behavior byte-for-byte. Realistic = starter deck ± random removals/additions (inclusive
      ranges), additions weighted 60/25/12/3 own/colorless/curse/off-character via `CardCatalog` (reads the
      game's real `ModelDb`/`CardPoolModel` pools — no hand-maintained lists); never deals status; added
      cards unupgraded; decks stay byte-identical for a seed. Observation `info`
      gains `deckSpec` kind + realistic `removedCards`/`addedCards`. Python `Lts2Env.reset_combat` takes a
      pass-through `deck_spec` dict (stdlib-only). Seeded determinism + bounds + no-status + explicit/absent
      parity covered by `DeckSpecTests`. _Product update:_ realistic now also grants `relics` `[0,2]` +
      `potions` `[0,1]` (HP-restoring/granting potions excluded — Blood Potion, Fruit Juice, Regen Potion,
      Fairy in a Bottle — via `PotionCatalog`), and varies the starter relic per fight (`starterRelic`
      `{absent:0.10, orobas:0.10}`: absent / Touch-of-Orobas upgraded+granted / normal); `info` gains
      `addedRelics`/`addedPotions`/`starterRelicState`/`upgradedStarterRelic`, and `StarterHeal` follows the
      starter-relic state. Sampled after the deck build so decks stay seed-stable.
- [x] **1.2 Catalogs** — _done._ `--dump-powers` mirrors `--dump-cards` (per power: id, Buff/Debuff type,
      stack/instance type, allowNegative, varKeys). `--dump-cards` extended with `rarity`, `pool` title,
      `category`, and `colorless`/`curse`/`status` flags (from the shared `CardCatalog` classifier the
      realistic sampler uses).
- [x] **1.3 Transition collector + corpus store** (contract 4) — _done_ (`python/lts2_agent/corpus.py`,
      `collect.py`, `corpus_report.py` + `tests/test_corpus.py`). `corpus.CorpusWriter` writes sharded
      gzip-JSONL under a root (default `python/data/corpus/`, gitignored) — one contract-4 record per
      decision (`{seed, scenarioMeta, t, state, options, actionTaken, nextState, nextOptions,
      rewardComponents, done, info}`, raw lossless wire observations; `rewardComponents` are raw
      before/after HP/block/enemy-HP with no reward function applied). **Leak-proof split**:
      `split_for_seed = crc32(fight seed) % 100` → 0-89 train / 90-94 val / 95-99 test, one function used by
      writer and reader, so a fight seed can never appear in two splits (unit-tested for determinism +
      disjointness). `python -m lts2_agent.collect` drives N parallel envs (thread pool, like the trainers)
      over mixed regimes (`broad`=deckSpec random / `realistic` / `mixed` 50-50) × mixed policies
      (`random` uniform-legal / `heuristic` + navigator for choices / `mixed`) × characters × acts,
      recording combat **and** `Choice` decisions; a fight is written atomically at its end and dropped
      cleanly (records discarded, logged, env recreated) on env error or the ~90-decision cap. Seed
      discipline enforced structurally: fight seeds are `CORPUS-<run-label>-<env>-<counter>`, the `PROBE-`
      namespace is refused, and `explicit` deckSpecs (closed-eval) are refused by the writer. Collection
      streams to the dashboard as a `kind="collect"` run (`collect.transitions_total`/`fights_total`/
      `errors_total`/`transitions_per_s` aggregates + per-fight `fight.won`/`fight.hp_lost` tagged
      act/room/character/regime/policy). `corpus_report` renders the CP2 artifact (composition + win-rate by
      split/regime/policy/act/room/character; realistic removal/addition histograms; realized vs configured
      60/25/12/3 added-card pool distribution; top-20 additions; a 20-deck sample; a seed→split determinism
      note) as text or `--json`. Original item: mixed policies (random, heuristic, current PPO) × mixed
      regimes (broad + realistic) × all characters × acts; target ~1M transitions to start. Collection
      progress/composition visible on the dashboard.
- [x] **1.4 Oracle prober** — _done_ (`python/lts2_agent/oracle.py` + `tests/test_oracle.py`). A
      **probe** freezes a reproducible combat position — `{probeId, resetParams, actionPrefix, meta}`
      reached by replaying an action prefix from a seeded `reset_combat` (no mid-combat snapshots exist).
      Three CLI commands: `build` (freeze a probe set, reproducible from `--master-seed`, spanning acts
      0-2 / monster-elite-boss room mix via the pct knobs / all characters / 0-15 step depth), `run`
      (replay each probe → ground-truth next observation for **every** legal action; gzip-JSONL shard,
      one record per probe; `--envs N` parallel over host processes), and `verify` (double-replay
      determinism spot-check — doubles as CP2's check). **Eval-only by construction:** every probe seed
      is `PROBE-`-prefixed, a reserved namespace training collectors must never use (`validate_probe_seed`
      / `assert_not_probe_seed` enforce it). **Reproducibility gate:** ~5% of deep-replay fights are
      genuinely non-reproducible (unreseeded RNG / async-pump ordering), including *flaky* ones that only
      diverge occasionally; the builder replays each candidate 8× on a second independent host and keeps
      only byte-identical ones, so every committed probe is a stable cross-process position. Committed
      `data/probes.json` is a light 40-probe set; the few-hundred-probe set is built at CP2. Env errors
      (per-action, per-probe, host crashes) are tolerated and recorded, never fatal. Original item: freeze
      a probe set (a few hundred positions spanning acts/rooms); replay-based ground-truth next states for
      every legal action at each probe. Eval-only.

**CP2 (manual review):** a dashboard page (or report) showing generated-deck distributions —
removals/additions histograms, pool-weight realization, a sample of 20 decks to eyeball for
"looks like act 1"; corpus composition stats; determinism spot-check (same seed twice ⇒ identical
deck and fight).

### M2 — Tokenizer (design P1)

- [x] **2.1 Entity tokenizer** (design §4.1) — _done._ `lts2_agent.tokens` encodes a state as a set of
      typed entity tokens (global / card / creature / power / intent / orb / relic / potion / pending),
      each with a token-type id, catalog/enum indices, and **symlog** numerics (±`NUM_CLIP`=1e5 clamp,
      exactly invertible for integer game quantities). `lts2_agent.catalog` generalizes `card_catalog` to
      all four dumped kinds (cards/powers/relics/potions) — stable dense id→index (0=none), static
      multi-hot table, content signature, CRC32 hashing fallback when a dump is absent; C# gained
      `--dump-relics`/`--dump-potions` mirroring `--dump-powers`. **Draw pile (and every card zone) is an
      unordered multiset** — card tokens are sorted by content so shuffle order can't leak (unit-tested).
      `TOKENIZER_VERSION` (=1) + the four catalog signatures are exposed for corpus/checkpoint/protocol
      stamping (contract 6). `coverage_check`/`detokenize` + the `--check` CLI ran over the corpus with
      **0 lost fields and 0 round-trip mismatches** across 160k states (80k-record `--check` pass).
      Waivers (in `tokens.WAIVERS`, reasons in code): non-combat room views
      (map/rewards/bundleChoice/event/shop/restSite/treasure/crystalSphere), `seed`/`netId`, run `deck`,
      static `poolId`; monster/character/orb/enchant/affliction ids + granted keywords are covered-lossy
      (hashed). **Padded dims (measured max over the full 1.0M-record corpus → cap):** cards 82→200,
      creatures 8→12, powers 24→96, intents 7→32, orbs 9→16, relics 8→24, potions 5→8. README
      "Tokenizer" section documents the contract. (2.2 PPO-on-tokens sanity pass now landed.)
      **v2 (2026-07, `TOKENIZER_VERSION=2`) — count-grouped card tokens:** identical-content card
      instances within a zone collapse to one token carrying an integer `count` (symlog; trailing
      `CARD_NUM` column), cards differing in any field stay separate; `detokenize` expands counts so the
      canonical dict (and every `statefmt`/`legal_actions`/report consumer) is byte-unchanged. Re-measured
      over the full 2.0M-state corpus: grouped card max **42** (v1 instance max 82), mean 14.11 instances →
      10.85 grouped tokens (1.30× shorter sequence); cards padded cap **200→64** (>3× smaller). Round-trip/
      coverage contract still 0 lost / 0 mismatches over the `--check` pass. Report footprint re-measured for
      v2 (`ACTION_FOOTPRINT` 0.1303→**0.1704**; `python -m lts2_agent.wm.footprint`).
- [x] **2.2 PPO-on-tokens sanity pass** — _done (CP3: comparison overlapped baseline; stopped early
      by product decision — encoding parity confirmed)._ `lts2_agent.model_tokens`
      is a set-transformer actor-critic over the tokenizer (per-token-type embedders with a **shared card
      embedder** for state cards and option cards; creatures fold in powers/intents by scatter-add then
      self-attend; learned latent queries attention-pool the token set into a state context `z`; ~1.3M params
      at `d_model=160`). Options are scored as (kind ⊕ option card/potion embedding ⊕ the **target creature's
      embedding gathered by `targetCombatId → creature slot`** ⊕ `z`) under a masked softmax, with a
      tanh-bounded ±20 value head — same PPO head shape as `model_torch`. Checkpoints stamp
      `tokens.tokenizer_signature()` and reject a mismatch loudly. The rollout (`rollout_torch`) and PPO update
      (`ppo_torch`) are shared with the features baseline via a defaulted `adapter` seam (`adapters.py`), so
      `train_torch` is byte-for-byte unchanged; the new trainer `train_tokens` reuses the same `ScenarioConfig`
      knobs + reward + fixed-seed greedy/sampled eval and streams a `kind="ppo-tokens"` metrics run for
      dashboard overlay. Serve path: `policies.torch_tokens_policy` (sampled-by-default). Tests in
      `tests/test_model_tokens.py` (forward shapes/masking, always-legal sampling, targeted-option→slot
      mapping, card-featurization parity with the tokenizer, checkpoint version-stamp rejection, serve
      parity/decline). A 25-iter GPU short run (default scenario settings) was healthy: no NaN, entropy stable
      ~1.2 (no collapse), explained-var rose 0.00→0.20, sampled eval win 0.75 (matches the baseline) / greedy
      0.25→0.50, sps ~200-290. **Remaining:** the full 300-iter baseline-comparison run (M0.5 bar: greedy 0.38
      / sampled 0.75) is the orchestrator's to run. Original item: attention encoder under the existing PPO
      head, trained on the realistic regime. Banks the model-free upgrade (design §6.A) and shakes out the
      tokenizer end to end before anything depends on it.

**CP3 (manual review):** round-trip/coverage report over the corpus (target: 100% of fields
accounted for); if 2.2 ran — dashboard comparison of PPO-on-tokens vs the M0.5 baseline on the
fixed-seed eval (expectation: ≥ baseline).

### M3 — Encoder + decoder (design P2)

- [x] **3.1 Encoder/decoder training** — _done (full 50k training run pending — the orchestrator's to
      run; this landed the module + a 4k-step verification run)._ `lts2_agent.wm` (`spec`/`encoder`/
      `decoder`/`model`/`report`/`data`) is a **set-transformer encoder → SimNorm latent `z` → symbolic
      decoder** trained supervised on the corpus (both `state` and `nextState` of every train-split record,
      ~2M states, streamed through a shuffle buffer + prefetch thread — no reward, no env). Encoder:
      per-token-type projections into `d_model=256` (cards/powers/relics/potions gather their static-catalog
      row) + type embedding → 4 pre-norm self-attention layers over the packed masked token set → Perceiver
      attention-pool into `z_dim=512`; **SimNorm** (groups of 8, TD-MPC2 §11 delta) makes `z` a concatenation
      of probability simplices (bounded, anti-collapse). Decoder: `z` → memory tokens → per-type learned slot
      queries (cross+self-attention) → per-type heads emitting the tokenizer's array space directly —
      categorical CE per `*_idx` column, **MSE on symlog `*_num`**, per-slot presence BCE, keyword BCE;
      canonical reconstruction reuses `tokens.detokenize` verbatim. ~10.1M params. The field spec both sides
      iterate lives in `wm/spec.py`; checkpoints stamp `tokenizer_signature()` and reject a mismatch.
      Trainer `train_encdec` (AdamW + warmup/cosine, `--steps`/`--val-every`/`--resume`/`--run-label`, fixed
      cached val sample) streams a **`kind="wm-encdec"`** metrics run: per step-window `train.loss`/
      `loss_categorical`/`loss_numeric`/`loss_presence`/`lr`/`states_per_s`; per val pass the per-field
      report card (`eval.card_id_top1`, `card_zone_acc`, `power_id_top1`, `power_amount_mae`,
      `creature_hp_mae`, `creature_block_mae`, `intent_damage_mae`, `energy_acc`, `relic_set_f1`,
      `potion_set_f1`, `hand_size_acc`, `pile_size_acc`, `pending_choice_acc`, aggregate
      `exact_state_rate`) — MAEs in RAW units — each emitted a second time tagged `{act}` for the dashboard
      group-by. `eval_encdec` prints the full-split report card (the CP4 artifact). Tests in
      `tests/test_wm_encdec.py` (forward shapes, SimNorm normalization, overfit-one-batch loss drop,
      report-card contract + detokenize hand-off, exact-state=1 on teacher-forced targets, checkpoint stamp
      rejection). **Speed:** on-the-fly Python tokenization put tokenization on the trainer's critical path
      (and it competed for CPU when the GPU was shared with another job); `lts2_agent.wm.cache build` now
      writes a one-time pre-tokenized `.npz` shard cache (multiprocessing pool; both `state`+`nextState`
      kept, no dedup, for exact distribution parity; signature-stamped manifest; auto byte-equality
      `--verify`). 2.0M states → ~179 MB in ~39 min. `train_encdec` reads it automatically when
      present+matching (loud error on mismatch). Measured (RTX 3090): cache data path **~7400 states/s** (vs
      ~960 on-the-fly single-thread), so training is now **fully GPU-bound** — the model forward+backward
      itself caps this box at **~470 states/s** (fp32, no flash-attention on Windows), ~11-12 h for a 50k×384
      run (the earlier ~30 h estimate was under GPU contention with a concurrent job). Reaching the
      >2000 states/s ceiling further is a GPU-compute problem (TF32/AMP, flash-attention, or a faster GPU),
      not a data-path one. Cache tests in `tests/test_wm_cache.py`.
      **Latent-shape A/B (design §10, first bullet — the CP4 decision):** `--latent-mode flat|tokens`
      switches only the latent structure between the pool and the decoder. `flat` (default) is byte-identical
      to the above (pooled → flatten → `z_dim=512` SimNorm vector → re-expanded to memory tokens). `tokens`
      keeps `--latent-k` (default 16) latent tokens as the latent (no flatten, no `z_dim` projection; SimNorm
      per token; decoder consumes them directly as memory — removing the flatten-to-512 squeeze suspected of
      capping card-identity reconstruction over big multisets). Dropping the two projections makes `tokens`
      the smaller model (~7.0M vs ~10.1M params). Checkpoint meta stamps `latent_mode`/`latent_k`; load
      rejects a mode mismatch loudly; the M4 predictor reads `latent_mode` to shape its latent. Tokens-mode
      tests added to `tests/test_wm_encdec.py`. **A/B VERDICT (product owner, 2026-07-16): token-set wins
      decisively** — same-budget curves: tokens `state_dist` 0.082 at 12.5k steps vs flat's 0.098 at 21k
      (better with 40% less compute, 31% fewer params), and a steeper power law (b≈0.54-0.64 vs 0.36);
      run stopped at ~19k on the owner's call. **`--latent-mode tokens` is the latent contract for M4.**
      Follow-on experiment series (5k-step probes vs the tokens control curve, one change at a time,
      `--halt-step` keeps the shared cosine-to-50k schedule): count-grouped card tokens (tokenizer v2),
      decoder-heavy scale, LR sweep + EMA, two-hot numeric heads, class-balanced card CE. Cross-tokenizer
      comparisons use `eval.action_snr` (footprint re-measured per tokenizer version), not raw
      `state_dist`. **Tokenizer v2 (count-grouped cards) shipped** (`TOKENIZER_VERSION=2`): card `MAX_CARDS`
      200→64, `CARD_NUM` gains a symlog `count` column, canonical dict byte-unchanged (all consumers
      untouched), and `ACTION_FOOTPRINT` re-measured 0.1303→0.1704 via `python -m lts2_agent.wm.footprint`.
      A v2 cache is a fresh dir (`--out data/corpus_tok_v2`); the v1 cache/checkpoints reject on the
      signature bump (correct). **(Superseded by tokenizer v3 — factored population rows with a per-zone
      count vector — see M3.5.)** **Probe flags shipped (default OFF, byte-identical when off, compose independently):
      `--num-head twohot` (64-bin symlog two-hot numeric heads), `--card-ce balanced` (1/sqrt(freq)
      card-identity CE, signature-cached), `--ema DECAY` (weight EMA; val + `.pt.ema` checkpoints);
      tests in `tests/test_wm_encdec.py`.**
- [x] **3.2 Decoded-state pretty-printer + diff view** — _done._ `lts2_agent.statefmt`:
      `format_state` renders any **canonical dict** (`tokens.detokenize` output — a decoder's output, or
      `detokenize(tokenize(raw wire))`) as compact text (player/Osty/enemies with hp/block/powers/intents,
      energy/stars/turn, hand with per-card cost/dmg/block/upgrade, draw/discard/exhaust as counted
      multisets, relics, potions, pending choice); `diff_states` is the field-level "what changed" view
      (HP/block/energy deltas, per-zone card multiset moves, powers gained/lost/changed, enemies died,
      intents changed) the TUI inspector (4.4) + report card (4.3) reuse. Hashed-lossy ids
      (monster/character/orb/enchant/afflict/keyword — `tokens.LOSSY_FIELDS`) resolve to names via an
      optional reverse map; `build-hash-names` CLI scans the corpus once → `data/hash_names.json`
      (**1.0M records → 120 buckets across 6 vocabs, 13 colliding buckets, all monster**), printer shows
      names when present else `#bucket`. Tests in `tests/test_statefmt.py` (synthetic render + moved-card /
      HP / new-power / enemy-died diffs + real fixtures).
- [x] **3.3 Legal-action derivation** — _done._ `lts2_agent.legal_actions.derive_option_keys` implements
      `GameHost.ListOptions` over **tokenized fields** (each hand card's `canPlay` + targetType × live
      hittable enemies → PlayCard-per-target; potions by catalog usage/targetType; EndTurn in combat;
      pending choice → SelectCards), scored as set-F1 vs the recorded options by option identity
      (kind + cardId/potion + targetCombatId; order-agnostic). CLI
      `python -m lts2_agent.legal_actions --corpus data/corpus --split val` prints overall + per-kind +
      per-phase rates + top mismatch patterns. **Measured on TRUE states (the upper bound), 47.4k val
      records: exact-set 99.82%, precision 0.99935 / recall 0.99937 / F1 0.99936** (PlayCard F1 0.9998,
      EndTurn 0.9999, Use/DiscardPotion 1.0000, SelectCards 0.9884). The residual is **two enumerated
      missing-information findings** (tokens NOT patched — reported per instructions): (1) the offered-card
      **order** for multi-select (`minSelect>1`) choices is lost by the sorted-multiset tokenization, so the
      game's exact-minimum SelectCards shortcut can't be reproduced (~4% of Choice records, 142 keys);
      (2) a post-combat **reward screen** whose wire `phase` is still `Combat` (`PendingRewards` isn't
      tokenized — rewards view is waived) derives combat options instead of TakeReward/Proceed (~14
      records, 143 keys). Tests in `tests/test_legal_actions.py` (synthetic rules + real fixtures derive
      the recorded set exactly, incl. via token round-trip).

**CP4 (manual review):** held-out reconstruction dashboard (~exact expected); a session with the
pretty-printer on random held-out states — do decoded states read as *the same fight* to a human?

### M3.5 — T3 factored architecture: tokenizer v3 (the data layer for the expert-per-category redesign)

- [x] **3.5 Tokenizer v3 — factored population rows + per-field ranges** — _done (tokenizer + spec-level
      range data + cache path + coverage/round-trip PASS + tests; the follow-up agent rebuilds
      `wm.encoder`/`decoder`/`model` experts on this spec)._ Agreed with the product owner as the data
      layer for a factored "expert-per-category" autoencoder redesign. Two changes to
      `lts2_agent.tokens` (`TOKENIZER_VERSION=3`):
      1. **Card population rows.** `zone` leaves the card grouping key. There is now **one row per
         distinct card CONTENT** (catalog id + every live dynamic field + keywords) carrying a **per-zone
         count vector** `count_{hand,draw,discard,exhaust,offered}` (`CARD_NUM` tail; `zone` removed from
         `CARD_IDX`) instead of a single `count`. Population membership is structural: a card moving
         hand→discard is the *same* row with the count shifting between two columns (the predictor
         expresses zone transitions as count arithmetic; creation/transform as rows appearing/
         disappearing). Cross-zone live-field divergence (a cost-reduced copy in hand vs its twin in draw)
         naturally splits into separate rows. `detokenize` expands the vector back to per-instance-per-zone
         canonical dicts, so the **canonical dict is byte-identical to v1/v2** — `statefmt`/`legal_actions`/
         `corpus`/report consumers untouched (verified by their green suites and the `_canon_dist`
         cross-version metric staying schema-identical).
      2. **Per-field integer ranges.** Every numeric column gains a measured `(lo, hi, resolution)` range
         in `wm/spec.py` (`NUMERIC_RANGES` + `RangeSpec` + `clamp_to_range`), scanned by a new streaming
         CLI `python -m lts2_agent.wm.ranges` (footprint's pattern; `--shard-stride` for cheap full-corpus
         breadth). These are the exact per-field domains a future per-field decoder bins against; the
         tokenizer keeps symlog storage (cache/decoder compat, exact `round(symexp)` round-trip), and
         out-of-range values clamp **loudly** (documented). `wm/spec.py` restructured for the zone-count
         columns + ranges while keeping `TYPES`/`TypeSpec` recognizable; `report.card_zone_acc` redefined
         to score the whole per-zone count vector; `model_tokens` (PPO) card embedder dropped the zone
         categorical column (its featurize is layout-driven, so otherwise unchanged).
      **Measured** (shard-strided 336k-state scan of `data/corpus`, `--shard-stride 12`; corpus2 was still
      collecting, so ranges/maxima used data/corpus with generous slack + loud clamping): population rows
      mean **10.21/state** (vs 14.21 instances → **1.39× shorter** sequences), rows **max 32** (v2
      zone-scoped grouped max 42, v1 instance max 82) → `MAX_CARDS` stays 64. Interesting ranges: energy
      0..40, per-zone counts 0..40, gold 0..5000, power amount −30..250, creature HP capped 0..1000 (the
      game's `999999999` sentinel clamps loud); `act`/`floor`/`ascension`/`score` widened past this act-0
      corpus's homogeneity (re-measure on corpus2). **v3 `ACTION_FOOTPRINT` re-measured 0.1704→0.1224**
      (`wm.footprint`, 3k val transitions; PlayCard median fell 0.141→0.050 because a play now mostly
      shifts counts between two columns of one row). Coverage CLI over the corpus: **0 lost fields, 0
      round-trip mismatches**. Tests added to `tests/test_tokens.py` (population grouping; same card in 3
      zones → one row with zone counts; zone-count expansion exactness; cross-zone divergence → separate
      rows; shuffle invariance under grouping; version==3; range-spec presence + loud clamping) and
      `tests/test_wm_encdec.py` updated for the v3 card spec. A v3 cache is a fresh dir
      (`--out data/corpus_tok_v3`); v1/v2 caches/checkpoints reject on the signature bump (correct).

- [x] **3.6 Factored expert autoencoder (`--arch factored`)** — _done (`wm/experts.py` +
      `wm/model_factored.py` + trainer/report/dashboard wiring + `tests/test_wm_factored.py`; the
      monolith `--arch mono` path is untouched/byte-identical)._ The T3 "expert-per-category" AE the v3
      tokenizer was the data layer for. The state latent is the **concatenation of per-expert slices** —
      a named, offset-addressable layout the M4 predictor will read/write by slice — with **no
      cross-category attention inside the AE** (independence is deliberate; cross-category coupling is the
      predictor's job). Three tiers:
      1. **Tier-1 scalar codec (`ScalarCodec`, parameter-free).** The global token (3 enum categoricals +
         14 numerics) and pending (4 numerics) encode to a deterministic slice — one-hot for the small
         enums, fixed **binary bin codes** (`NUMERIC_RANGES` bin index) for the numerics. Encode and
         decode are both fixed functions, so the round-trip is **exact by construction** for any in-range
         integer with *no learned weights*: `eval.scalar_exact` is 1.0 at step 0 (the wiring canary). The
         only misses on real data are the documented loud clamps (the `maxSelect`/HP `999999999`
         sentinels): 0.9999 on corpus2 val = 26/24000 choice states carrying the no-limit `maxSelect`.
      2. **Tier-2 small experts.** creatures (folds powers + intents into one set expert, parent-slot
         embedding kept), relics (multi-hot **set-membership** head, duplicate-free by construction —
         ported from the monolith's `--relic-head set`), potions (per-slot categorical — potions can
         duplicate), orbs. The small single-type experts (relics/potions/orbs) run at 1 enc / 1 dec layer.
      3. **Tier-3 card-population expert.** the largest slice; a set enc/dec over the v3 population rows.
      **All learned numerics decode through per-field range-bin classification** (`RangeBinHeads`) instead
      of the monolith's shared symlog MSE — creature HP gets resolution-1 bins over `[0,1000]`, killing
      the ±1 rounding tail — while still emitting the identical symlog `num` block (argmax bin → integer →
      symlog) so `reconstruct_arrays`/`report` consume factored outputs **unchanged**. Every expert's
      encoder carries an always-valid sentinel token so an empty category (no orbs/potions) never
      produces a fully-padded-attention NaN. **Metrics:** all existing report-card metrics flow;
      **`eval.expert_dist`** (per-expert share of `state_dist`, emitted tagged `{"expert": …}`, partitions
      the whole *exactly* — the `_state_dist` den-floor was removed so an empty category contributes den 0
      not 1) + **`eval.scalar_exact`** are new (dashboard `METRIC_LABELS` + `BOUNDED_01` updated).
      **Slice layout** (`d_model=256` defaults): `scalars 116 · creatures 768 · cards 1536 · relics 512 ·
      potions 128 · orbs 128` → **latent_dim 3188** (~ the monolith's 4096 tokens-mode budget); **22.9 M
      params** (cards 7.9 M largest → creatures 6.4 M → relics/orbs/potions ~2.5–3.5 M → scalars 0), vs the
      monolith's 10.1 M (flat) / 6.9 M (tokens) — independence replicates enc/dec machinery per expert,
      the cost of the clean per-slice seam. Checkpoints stamp `arch=factored` + the slice layout; loads
      reject a non-factored or layout-mismatched checkpoint. **GPU sanity** (RTX 3090, 450×384, bf16, temp
      v3 cache): losses fall (train 7.3→2.1, val 8.4→2.0), `scalar_exact`≈1.0 and `energy_acc`=1.0 from
      the first val, `expert_dist` tags flow; **faster than the monolith — ~1.69 k states/s vs ~1.50 k**
      (~12 %) at equal batch/AMP, the smaller per-expert attention scopes winning despite the extra
      per-expert kernel launches. Longer training (the report card / M4 gate) is the orchestrator's job,
      not this slice.

- [x] **3.7 Per-expert training infrastructure + relic bake-off** — _done (`train_encdec`
      `--train-experts`/`--val-experts`/`--init-expert-from`; `RelicExpert.relic_head` set/slots;
      per-expert checkpoint stamps + `wm/compose.py`; `eval.expert_exact`; `eval_encdec` composite-
      transparent; `tests/test_wm_factored.py` +8 → 21; README per-expert workflow)._ The
      product-owner's **sequential per-expert strategy**: because the experts are parameter-disjoint,
      train one until its slice hits a high exactness bar, keep it, iterate on the weak ones — never
      retrain a healthy expert because another struggles.
      - **Freeze/skip** (`--train-experts a,b,…`, factored only, default all): non-listed experts are
        excluded from the optimizer (byte-identical after steps — tested) *and* their encode/decode is
        skipped in the step (loss only over trained experts). A **solo relic run hit ~12.8 k states/s**
        vs the joint run's ~1.5 k (**~8.5×**), trainable params 3.55 M of 22.9 M.
      - **Focused val** (`--val-experts trained-only`): reconstructs only the trained experts' token
        types (no full report-card decode), emitting their `expert_dist`/`expert_exact` (+ relic F1);
        `.best` driven by the trained experts' mean `expert_dist`.
      - **`eval.expert_exact`** (new, tagged `{"expert": …}`): per-expert fraction of val states whose
        slice-owned token types reconstruct exactly (array-space, integer-rounded, presence incl.) — each
        expert's "done" bar. Partitions cleanly (tested: `expert_exact == (expert_dist_num == 0)`; scalars
        pins to 1.0 by construction).
      - **Per-expert checkpoint + compose**: each factored checkpoint's meta stamps a per-expert block
        (slice layout + tokenizer signature + build kwargs); weights live in the one full `state_dict`
        under `experts.<name>.*`. `python -m lts2_agent.wm.compose --out C --base B --experts
        relics=A.best …` assembles a standard factored checkpoint from per-expert sources, validating
        shared-global-config + per-slice width/config match. `eval_encdec` auto-detects `arch=factored`
        and loads composites transparently. **`--init-expert-from name=ckpt`** warm-starts one expert's
        slice from a full checkpoint (seed a solo run from the joint run's progress).
      - **Relic bake-off** (each a fresh solo relic run, 6k steps, seed 0, corpus_tok_v3 val 3000,
        batch 512, warmup 800, cosine). Final step-6000 val (matched steps):

        | variant | config | expert_dist ↓ | relic_set_f1 ↑ | expert_exact ↑ |
        |---|---|---|---|---|
        | a. set+count, pw 1     | `--fac-relic-head set --relic-pos-weight 1`   | 0.906 | 0.365 | 0.120 |
        | a. set+count, pw 5     | `--relic-pos-weight 5`                        | 0.893 | 0.370 | 0.120 |
        | a. set+count, pw 15    | `--relic-pos-weight 15`                       | 0.877 | 0.415 | 0.220 |
        | b. set+count, deep dec | `--relic-pos-weight 5 --relic-dec-layers 3`   | 0.883 | 0.453 | 0.220 |
        | c. set+count, lr 1e-3  | `--relic-pos-weight 5 --lr 1e-3`              | 0.920 | 0.328 | 0.109 |
        | **d. slots + dedup**   | `--fac-relic-head slots`                      | **0.645** | **0.657** | **0.334** |

        **Winner: (d) slots + inference dedup — decisively, on all three metrics** (expert_exact 0.334 vs
        the best set-head 0.220; relic_set_f1 0.657 vs 0.453; expert_dist 0.645 vs 0.877). The set head
        does **not** beat slots+dedup at matched budget; higher `pos_weight`/deeper decoder help it only
        marginally (pw 15 / deep tie at exact 0.220) and higher LR hurts. **Per the owner's standing rule
        — the set head ships only if it BEATS slots+dedup — the recommendation is to revert the factored
        relic expert to `--fac-relic-head slots` (the monolith path that reached 0.995 F1 with longer
        training).** These 6k-step solo numbers are all still below that bar (slots needs more steps to
        converge), but the *ranking* is unambiguous. Solo states/s across the sweep: **~12.0 k median
        (10.8–13.3 k)** — ~8× the ~1.5 k joint run, confirming the freeze/skip win.

- [x] **3.8 Solo-run dynamics fixes + the collapse bug** — _done (`wm/experts.py` LayerNorm-before-SimNorm;
      `--num-targets twohot|hard`; `--focus-present R` + `wm/data.focus_present_batches_cpu`;
      `wm/overfit.py` gate; `tests/test_wm_factored.py` +6)._ The product owner measured slow, spiky,
      **non-log** solo curves (orbs improved only 0.40→0.27 over 12 k steps despite being far simpler than
      cards). Building the **overfit-one-batch gate** (`python -m lts2_agent.wm.overfit`) surfaced the root
      cause: **SimNorm representation collapse**. The learned experts' `to_slice` Linear is unbounded, so
      training runs its magnitude away (measured: pre-SimNorm latent std 0.1 → **217** across states) and
      the grouped softmax **saturates to a state-INDEPENDENT one-hot** whose gradient vanishes — every
      distinct state encodes to the *same* latent (post-SimNorm std → exactly **0.0**), so no expert could
      overfit even a handful of states (orbs stuck at `expert_dist` 0.77, creatures 0.65 — dead flat). The
      existing `test_..._overfits_one_batch` only asserted a 40 % loss drop, so it never caught this.
      **Fix: a `LayerNorm(elementwise_affine=False)` before SimNorm** in `SetExpert`/`RelicExpert` — it
      bounds the logits so SimNorm stays sensitive to the state (output is still a concatenation of
      probability simplices, SimNorm's contract; no affine so weight-decay can't shrink a scale back toward
      the uniform-softmax collapse; no new state-dict keys). **Overfit gate after the fix** (batch 256,
      ≤2500 steps, twohot; pre-fix all were dead-flat at the noted floors):

      | expert | pre-fix (flat) | post-fix final `expert_dist` | steps→<0.01 |
      |---|---|---|---|
      | relics    | ~0.6 | **0.0011** | **1750** |
      | potions   | ~0.6 | **0.0192** | (≈, more steps) |
      | creatures | 0.646 | **0.0331** | capacity/steps |
      | cards     | ~0.6 | 0.153 | capacity/steps |
      | orbs      | 0.77 | 0.170 | capacity/steps |

      relics **passes**; every expert now **descends steeply** instead of flatlining (a fixed batch of 8
      distinct orb states overfits to `expert_dist` 0.0 — impossible pre-fix). The rich/multi-item cases
      (cards, orbs) need more than 2500 steps at batch 256 to cross <0.01 — a per-state bottleneck-capacity
      limit, not a wiring bug. **Numeric decode is argmax**, not expectation: the gate measures both and
      argmax's exact-bin rate is ≥ expectation's on every numeric type (e.g. orb 0.927/0.850, card
      0.908/0.845) — argmax lands the exact bin; a soft expectation straddles and rounds to a neighbour.
      Two further knobs (each independently toggleable; twohot default-on, the rest opt-in):
      - **`--num-targets twohot`** (default): distance-aware symmetric triangular range-bin targets
        ({0.25,0.50,0.25}) restore the numeric metric structure the monolith's two-hot gave (the fine
        integer grid makes a literal two-adjacent-bin split a no-op, so a small kernel is the faithful
        generalization). Flag fields stay one-hot.
      - **`--focus-present R`** (solo only): oversample states with a present trained-expert token (orbs are
        present in only ~19 % of states, so ~81 % of a solo batch was empty slots); a bounded two-pool
        sampler draws R present + 1−R empty, and is a cheap no-op for dense experts (creatures/potions).
      **Streaming probe finding (orbs canary, 6 k steps):** the collapse fix is *necessary for the fixed-
      batch overfit / capacity* but does **not** by itself change the streaming orbs curve — pre-fix and
      post-fix both plateau ~0.31 with the same early lag (the hard all-or-nothing `expert_dist` sits at
      1.0 while the loss falls, then flips once the argmax decode becomes correct). The streaming plateau is
      governed by **presence sparsity + batch noise**, which `--focus-present` / larger batch target.
      **Streaming probe table** (orbs solo, `expert_dist` ↓ at the step; 6 k steps, val-every 250):

      | variant | recipe | ~1k | ~2k | ~4k | note |
      |---|---|---|---|---|---|
      | a0 pre-fix        | b384 hard, no focus            | 1.00 | 0.37 | 0.35 | stuck→plateau, spiky (the reported bad curve) |
      | a  fix baseline   | b384 hard, no focus            | 1.00 | 0.27 | 0.31 | ≈ a0 — the collapse fix does **not** move the streaming curve |
      | b  +twohot        | b384 twohot                    | 0.69 | 0.37 | — | knob alone: stuck ~as long as baseline, ≈ baseline plateau |
      | d  +big batch     | b1536 hard                     | 1.00 | 0.65@1.25k | — | knob alone: still stuck early, smoother |
      | c  **+focus**     | b384 hard, focus 0.9           | **0.27** | **0.18** | **0.12** | escapes ~2× earlier and keeps falling well below the baseline plateau |
      | e  **all**        | b1536 twohot, focus 0.9        | 0.23 | — | — | fastest escape (0.30 by step 750, 0.23 by 1k) |

      **Verdict:** the dynamics ARE fixed. The **collapse fix** is the correctness/capacity foundation (the
      overfit gate went from impossible to passing); **`--focus-present`** is what restores the *log-shaped
      early ramp* on the sparse streaming curve (twohot and big-batch alone do not — they stay stuck like
      the baseline). twohot's payoff is on the numeric-heavy experts (creature/card HP), not orbs' two small
      fields. The remaining floor at ~0.18 is orbs' presence-calibration + per-state capacity, the next lever.

- [x] **3.9 Synthetic-space training for the finite experts** — _done (`wm/synth.py` per-expert generators;
      `train_encdec --data real|synth|mixed:R`; seeded synthetic coverage-val emitted tagged
      `{expert, val:"coverage"}` beside the real val; `tests/test_wm_synth.py` +18; README + dashboard
      labels)._ **Product decision (owner):** a finite-space expert (potions / relics / orbs, and — as
      coverage insurance — creatures / cards) is decoupled from game data and trained on **synthetic
      uniform configurations generated mechanically in tokenizer-array space** — no cache/corpus once
      decoupled. Rationale: the decoder is the predictor's API and must decode **any valid configuration**,
      not just the game-frequent ones; uniform coverage kills the rare-tail floors game-frequency training
      leaves (measured: potions capped at `expert_exact` **0.995** by ~3 rare belt configs — non-left-packed
      belts, empty-slot interleavings), replaces `--focus-present`, and fixes relic frequency imbalance.
      - **Generators** (`wm/synth.py`, per expert, uniformly-with-design, seeded): every array convention
        preserved exactly — left-packed presence + zeroed padding; **potion index-0 = empty belt slot at any
        position** (non-left-packed + fully-empty belts); **relics positional with legal duplicates** (v5 —
        random ids at explicit slots 0..k−1 in generated order, duplicates injected, order preserved);
        catalog/enum/hashed id ranges; **symlog storage** inside the measured `spec.NUMERIC_RANGES` (reusing
        the tokenizer's encoding so the training target recovers the exact integer). Creatures fold
        powers/intents with **valid parent refs**; cards keep a **game-shaped population structure** (row
        count + per-zone count vector from measured marginals) with uniform ids/dynamic-numerics. **All
        canonicalized/positional types are emitted in canonical order** — the generator-canonicality guard
        (3.11) asserts synth reproduces the same invariant as `tokenize`.
      - **Trainer**: `--data synth` (generator batches, no corpus/cache) | `real` | `mixed:R` (fraction R
        synthetic per batch, 1−R real). Factored **solo** only. **Val always runs both yardsticks**: the
        real fixed val (deployment) AND a fixed seeded 2000-config synthetic **coverage val**, the latter
        emitted a second time tagged `{expert, val:"coverage"}`.
      - **Doctrine amendment (owner, 2026-07-18): synthetic-first / real-data-is-eval-only.** Training is
        synthetic-first across the board — `--data synth` is now the **DEFAULT for a factored solo run** and
        **no full-corpus token cache is built any more**. `--cache` defaults to empty; the real fixed val
        (~2k states) is tokenized **on the fly** each run (seconds) so the deployment yardstick always runs,
        and `eval_encdec` streams + tokenizes the corpus directly. `real`/`mixed:R` remain opt-in escape
        hatches (they can still use an explicit `--cache`). Next: **cards** get a full-synthetic attempt,
        with `mixed:R`/`real` as the cards-specific fallback only if synthetic-only underperforms.
      - **Probes** (6k steps, batch 512, seed 0, synth, `--num-targets twohot --loss-balance expert`; on-the-
        fly real-val). `expert_exact` ↑ on the real val (deployment) and the synthetic coverage val:
        <!-- PROBE_TABLE -->
      - **Verdicts:** <!-- VERDICTS -->

- [x] **3.10 Representational well-posedness — tokenizer v4 (`TOKENIZER_VERSION = 4`, "v3.1")** — _done
      (`tokens.py` + `wm/spec.py` + `wm/synth.py`; `tests/test_tokens_wellposed.py` new + updated
      potion/orb tokenize tests; README + this amendment)._ **Measured diagnosis:** each per-category
      expert is a **permutation-invariant set encoder**, so a per-slot target that varies with input order
      but is carried in **no per-token field** is ill-posed — the encoder maps every permutation of a
      multiset to the same latent while the targets differ. Proven cleanly: permuted potion belts encoded
      **byte-identically** while their per-slot targets differed, and a slice-width A/B (128 vs 256) moved
      the potions `expert_exact` **not at all** (pinned 0.458/0.459); real data masked it only because game
      belts arrive canonically packed. The fix is per-expert **by semantics**:
      - **Potions — canonicalize position away (left-pack).** Slot identity is decision-irrelevant (options
        key on potion id), so the belt is emitted **non-empty first (sorted by catalog index), then index-0
        empties**, preserving belt SIZE. `detokenize` emits the same layout, so a rare non-left-packed raw
        belt (`[empty, empty, STRENGTH]`) round-trips to its canonical `[STRENGTH, empty, empty]`. The
        canonical-dict `potions` list ORDER therefore changes to the left-packed form — an accepted
        canonicalization (id-based consumers `statefmt`/`legal_actions`/corpus are unaffected; every such
        test still passes). Measured in `data/corpus2`: **82** rare non-left-packed raw belts now
        canonicalize; potion-belt tokens are **20000/20000 permutation-identical**.
      - **Orbs — position is semantic (evoke order): add a positional categorical.** Each orb token gains a
        `slot` column (0..MAX_ORBS−1) the set encoder can represent; synth keeps the learnable positions
        (left-packed slot == index). **3762** orb-bearing player-states in `data/corpus2` exercise it.
      - **Creatures / relics — canonicalized (sorted by content).** Audit finding: creatures were emitted in
        WIRE order (player, osty, enemies-as-listed) and the set encoder cannot see slot position, while the
        decoder reconstructs per fixed slot and powers/intents join by parent-slot index — the SAME trap for
        multi-enemy fights. combatId already carries a creature's identity, so order is not independently
        semantic → sort by `(kind, combatId, identity, hp, maxHp, block, active)` (kind keeps
        player<osty<enemy; parents follow the sort automatically via the tokenize flatten). Relics are an
        order-free set → sorted (also aligns the target with the set-head's sorted decode). **Cards were
        already content-sorted (well-posed) and are unchanged** — confirmed by the permutation test.
      - **Well-posedness suite** (`test_tokens_wellposed.py`): for **every** variable token type, assert it
        is EITHER canonical-order-invariant to wire permutations (cards/creatures/powers/intents/relics/
        potions) OR carries an explicit slot-index column (orbs). The regression guard.
      - **Cache/footprint:** rebuilt into `data/corpus_tok_v31` (v4 signature invalidates v3 loudly);
        re-measured `ACTION_FOOTPRINT` on `data/corpus2` → **0.1105** (v3 was 0.1224). Round-trip stays
        exact (8000/8000 real corpus2 states).
      - **Canary probes** (6k steps, batch 512, val-every 50, `data/corpus_tok_v31`): <!-- WELLPOSED_PROBES -->

- [x] **3.11 Relics are POSITIONAL — tokenizer v5 (`TOKENIZER_VERSION = 5`, "v3.2")** — _done
      (`tokens.py` + `wm/spec.py` + `wm/experts.py` + `wm/synth.py` + `wm/model_factored.py` +
      `wm/model.py`/`decoder.py`/`report.py`; `tests/test_tokens*.py` + `test_wm_*` updated; README +
      this amendment)._ **Product-fact reversal (owner):** two v3.1/v4 assumptions about relics were
      wrong — (a) relic ORDER is semantic (wax relics such as Tezcatara via Toy Box expire in acquisition
      order, carried only by the wire's ordered id list), and (b) duplicate relics DO occur and the total
      can exceed the old 24 cap in long runs. Both invalidate v4's "sorted, order-free, unique" relic model.
      **Fix — the orb treatment:** relics become a POSITIONAL type — one token per relic INSTANCE, WIRE
      ORDER preserved (v4's `relics.sort()` reverted), each token carrying an explicit `slot` (== list index)
      the permutation-invariant relic expert can see and target; `detokenize` emits the ordered id list
      verbatim (canonical-dict `relics` returns to a flat wire-order id list — id-based `statefmt`/
      `legal_actions`/corpus consumers unaffected). `MAX_RELICS` 24→40 bounds TOTAL relics with a
      strict-overflow loud clamp (`data/corpus2` scan of 4.0M states: max total 8, max copies of one relic
      2, 3238 states carry a duplicate). **Deletions (simplification is part of the deliverable):** the
      multi-hot relic **set head** + cardinality head, `_dedup_slot_ids`, `_decode_set_head`, the
      `--relic-head` / `--fac-relic-head` / `--dedup` flags and their plumbing, and the bespoke
      `RelicExpert` — relics now ride the generic single-type `SetExpert` (positional per-slot categoricals
      + presence), identical to potions/orbs. The 3.7 set-vs-slots bake-off above is thus moot.
      - **Generator-canonicality guard (new):** the synth generators bypass `tokenize`, so a companion
        regression test asserts each generated type reproduces the SAME canonical invariant (positional
        types carry `slot == index`; canonicalized cards/creatures emit rows in the tokenizer's sort order).
        This closed the root cause of the earlier relic plateau — v3.1 synth wrote relic ids in random draw
        order against a canonically-sorted target (slots-synth F1 ~0.99 but exact ~0.10, the arrangement
        lottery). `synth._fill_relics` now emits positional rows (slots 0..k−1) with a small duplicate
        probability so duplicates are LEARNED as legal.
      - **Cache/footprint:** rebuilt into `data/corpus_tok_v32` (v5 signature invalidates v4/v3 loudly);
        re-measured `ACTION_FOOTPRINT` on `data/corpus2` → **0.1091** (v4 was 0.1105 — the relic `slot`
        column widened the field universe slightly).
      - **Canary probes** (synth training, on-the-fly real-val + seeded coverage-val, batch 512, seed 0,
        `--num-targets twohot --loss-balance expert`, 2026-07-18):

        | probe | steps | real-val exact | real-val relic_f1 | real-val dist | coverage exact |
        |-------|------:|---------------:|------------------:|--------------:|---------------:|
        | **relics** (target) | 6000 | **0.541** | **0.958** | 0.063 | 0.474 |
        | potions (regression) | 2000 | **1.000** | — | 0.000 | 0.922 |
        | orbs (regression) | 2000 | 0.807 | — | 0.515 | 0.052 |

        Relics is the headline: the positional representation + the generator-canonicality fix take real-val
        **exact from the old ~0.10 arrangement-lottery to 0.541** while holding membership quality (relic_f1
        0.958 ≈ the slots-level F1), and the curve is a smooth log-ascent (f1 0→0.94 by ~3.8k→0.958 at 6k) —
        no plateau. The remaining exact↔F1 gap is the per-row id+slot order-statistic cascade on larger
        sets, which the game-sized caps already shrink. Potions (the v4 well-posedness win) is unregressed:
        real-val exact a perfect 1.000, coverage 0.922. Orbs (untouched by v5) trains normally under the
        version bump — its low coverage-exact is the expected early state of a 2k-step run over its two
        wide-range numerics (evokeValue 0..250), not a regression.

### M4 — Predictor (design P3) — the heart, and the main research risk

- [ ] **4.1 Afterstate step**: K-step unrolled training with latent-consistency loss + SimNorm,
      reward-component and terminal heads (design §4.4 incl. the §11 deltas).
- [ ] **4.2 Chance step**: discrete codebook; End-Turn semi-supervision from logged draws;
      per-code perplexity and calibration metrics.
- [ ] **4.3 Prediction report card**: per-field accuracy × per-action-kind × K∈{1,3,5}, held-out
      corpus + oracle probes; dashboard panels + a printable per-run summary. This is the
      milestone's *product* — it must localize failures ("EndTurn intent prediction is weak in
      act 2 elites"), not just average them away.
- [ ] **4.4 TUI prediction inspector** (protocol v2, contract 5): select any option → decoded
      predicted next state (as a diff vs current); for stochastic actions, top-k outcomes with
      probabilities; after the action resolves, show predicted-vs-actual as a diff. This is the
      "watch the model think" feature and the fastest path to spotting rule misunderstandings.
- [ ] **4.5 Unleash/Osty acceptance set**: extend `closed_eval` scenarios to prediction checks
      (does the predicted damage of Unleash track Osty's HP? does Bodyguard's predicted state
      show the summon?). Evaluation-only, as always.

**CP5 (manual review):** the report card, plus a TUI session with the prediction inspector —
deliberately try to fool the model (multi-hit intents, AoE lethal, Discovery, X-costs) and file
what breaks. **Decision gate:** if deterministic-step accuracy is weak and un-fixable here, stop
and keep M0–M3 (design §9's fallback) rather than building planning on sand.

### M5 — Value head + greedy afterstate agent (design P4)

- [ ] **5.1 Value training** (TD on real fights, realistic regime; bounded HP-fraction target)
      with calibration panels (predicted vs realized final-HP-fraction).
- [ ] **5.2 Greedy/sampled afterstate policy** served through the decision server (v2 explanations
      include per-option values); collection switches to this agent on the realistic regime;
      corpus keeps growing (the predictor keeps training on the new data).
- [ ] **5.3 Comparative eval**: fixed-seed protocol vs the M0.5 PPO baseline and the heuristic,
      with act/room breakdowns.

**CP6 (manual review):** dashboard comparison (expectation: beats PPO baseline and heuristic on
fixed-seed eval, and the act/room breakdown shows *where* the wins come from); a TUI session
sanity-checking that ranked values agree with intuition on obvious spots (free lethal, lethal vs
overkill-block).

### M6 — Intra-turn planner (design P5)

- [ ] **6.1 Beam search** over deterministic card chains to the end-turn boundary; chance-action
      leaves by expectation/top-k codes; latency measured against the decision-server budget.
- [ ] **6.2 Policy prior + distillation** from planner output; sampled variant for collection.
- [ ] **6.3 Planner metrics**: regret vs oracle on the probe set; win-rate vs beam width
      (the scaling curve that says whether deeper search is worth it); TUI shows the planned
      line ("intends: Defend → Bodyguard → Unleash → End Turn").

**CP7 (manual review):** beats CP6 numbers; the scaling curve; TUI sessions on fights the M5
agent lost — does the plan view make the improvement (or remaining failures) legible?

### M7 — Contingent extensions (design P6)

- [ ] Gumbel chance-node MCTS across turns; Reanalyse; imagination training — **only** where
      CP7's scaling curves and the report card say the model supports it. Evaluate LightZero as
      a base/reference before writing MCTS from scratch (design §11).

---

## QA process (cross-cutting)

- **Automated, per landing** (the repo's normal bar): seeded deterministic tests on both sides
  (C# `dotnet test --filter`, Python unit tests); tokenizer round-trip tests; corpus split-hygiene
  test (no fight seed in two splits); protocol v2 golden-message tests; the fixed-seed eval
  runnable as one command for any agent (heuristic/PPO/world-model) so numbers stay comparable
  across the whole effort.
- **Continuous**: the dashboard *is* the QA surface for training — every learning component
  ships its metrics panel in the same PR that makes it train (the M0 rule). Training-side
  regressions are judged against recorded baseline runs, not memory.
- **Manual checkpoints CP1–CP7**: as specified per milestone — each names what to run, what to
  look at, and what "good" looks like. CP5 is additionally a go/no-go decision gate.
- **Anti-overfit discipline** (goal 4): probe sets, closed-eval scenarios, and oracle labels are
  evaluation-only and never enter a training corpus; training inputs are always sampled
  distributions (broad or realistic), never fixed instances. Enforced structurally: collectors
  refuse `explicit` deckSpecs.

## Known M4 consideration — draw-pile stacking vs the information-set boundary

Some cards place chosen cards ON TOP of the draw pile (e.g. Regent's Photon Cut, colorless
Thinking Ahead), making the next draw(s) deterministic *to the player*. The tokenizer's draw-pile
multiset canonicalization is correct as an anti-leak default (the agent must not see shuffle
order) but it also discards this *legitimate* knowledge — the target representation is the
player's INFORMATION SET (multiset + known placements), and "which cards are known" is a function
of history (stack effect since last shuffle), not derivable from one wire state. Plan when M4
lands: (a) within-plan knowledge (stack effect played inside the current search) is carried by
the predictor's recurrent latent through the unroll — design the predictor latent to support
short-horizon private state and add this as an explicit predictor test case; (b) across-the-root
knowledge needs either a harness-computed `knownTopCards` wire field (track pile manipulation
since last shuffle) or an accepted, MEASURED error — seed the oracle probe set with
positions immediately after stack-the-pile effects to quantify it before deciding; (c) the value
function absorbs residual EV error either way. Owner-flagged 2026-07-18.

## Sequencing notes & watch-items

- M0 and M1 are independent and can interleave; M2+ is strictly ordered. The TUI prediction
  inspector (4.4) can start as soon as M3's pretty-printer exists, against a stub predictor.
- Protocol-v2 payload size and decision latency need measuring at M4/M6 (predictions × ~20
  options; search inside the TUI timeout) — both have easy mitigations (summarized diffs,
  explain-on-demand for a single option) if they bite.
- The realistic-deck weights (60/25/12/3) and removal/addition ranges are product decisions, not
  constants of nature — put them in the `deckSpec`, surface them in run manifests, and expect to
  tune after CP2.
- Keep the PPO baseline runnable (don't break `train_torch`) until M5 has beaten it on record.
