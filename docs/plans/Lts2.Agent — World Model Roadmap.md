# Lts2.Agent — World-Model Roadmap (implementation backlog)

Status: **M0–M2 done (CP1–CP3 approved). Next: M3 encoder/decoder.** CP3 verdict: PPO-on-tokens
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
      rejection).
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
