# lts2_agent — Python interface to the headless STS2 emulator

`lts2_agent` lets you **train** an agent against the game harness and **run** that same agent from
the TUI, over one small JSON-lines protocol. No third-party Python dependencies (stdlib only); the ML
framework is your choice.

Two flows, one wire schema:

| Flow | Who drives | C# side | Python side |
|------|-----------|---------|-------------|
| **Training** | Python | `Lts2.AgentHost` (environment server) | `Lts2Env` spawns it, sends `reset`/`step` |
| **Evaluation** | C# TUI | `ProcessDecisionEngine` (client) | `decision_server` scores `evaluate` requests |

Because the observation (`state` + `options`) and the action encoding (an **index** into `options`)
are identical in both directions, a policy trained in the first flow plugs straight into the second.

## Prerequisites

- .NET 9 SDK (to build the C# host) and Python 3.10+.
- Build the environment host once:

  ```sh
  dotnet build src/Lts2.AgentHost/Lts2.AgentHost.csproj
  ```

## Training

```python
from lts2_agent import Lts2Env

with Lts2Env(seed="RUN1", character="Ironclad") as env:
    obs = env.reset()
    while not obs["done"] and obs["options"]:
        action = 0                     # an index into obs["options"]; plug your policy in here
        obs = env.step(action)
    print("final score:", obs["info"]["score"])
```

- `obs["state"]` — the full immutable game state (see `src/Lts2.Harness/GameState.cs`).
- `obs["options"]` — the legal actions; `env.step(i)` applies `options[i]`.
- `obs["info"]` — reward-relevant scalars (`score`, `floor`, `victory`, per-player `currentHp`/`gold`).
  **Reward is up to you** — the host never assumes a reward function.
- For a "choose N of M" card choice, pass a list of card indices: `env.step([0, 2])`.

Skeleton loop: `python -m lts2_agent.examples.train_stub --episodes 2 --seed DEMO`.

## Evaluation (from the TUI)

Point the TUI at a decision server via environment variables, then pick it from the **Strategy** menu:

```sh
# Windows PowerShell
$env:LTS2_AGENT_CMD = "python"
$env:LTS2_AGENT_ARGS = "-m lts2_agent.decision_server lts2_agent.policies.heuristic:policy"
$env:LTS2_AGENT_NAME = "Heuristic (py)"
dotnet run --project src/Lts2.Tui
```

The TUI launches the command, and each auto-play recommendation (the `[tab]` pick) comes from your
Python policy. If the process dies or misbehaves, the TUI simply shows no recommendation.

## Writing a policy

A policy is `policy(state, options) -> ranking`, returning either an `int` (the chosen index) or a
list of `(index, score)` / `{"index", "score", "rationale"}` entries (a subset is fine; empty =
decline). See `lts2_agent/policies/heuristic.py`. Load any policy with
`python -m lts2_agent.decision_server your.module:policy`.

**stdout is reserved for protocol messages** on both sides — log to stderr only.

## Learned policy: PPO combat engine (JAX)

A trainable neural **combat** policy lives alongside the reference heuristic. It decides only combat
(`PlayCard`/`EndTurn`, target included); the scripted `navigator` handles every non-combat phase so
runs complete. It is trained with PPO against the environment server and served back into the TUI
through the very same decision-server path — same features, same model.

Install the extra deps (isolated; the protocol/env modules stay stdlib-only):

```sh
python -m venv .venv && .venv/Scripts/pip install -r requirements-train.txt   # Windows
# or: python -m pip install -r requirements-train.txt
```

Pieces (all under `lts2_agent/`):

| Module | Role |
|--------|------|
| `features.py` | shared state/option encoders + a stable hashed card-id vocab — **the train/serve parity contract** |
| `model.py` | Flax actor-critic: a *per-option* scoring head (handles the variable, targeted action set) + a value head; plus checkpoint save/load |
| `reward.py` | reward shaping — run mode (HP + damage + kills + floor/win) and scenario mode (win/loss + HP lost) |
| `navigator.py` | scripted non-combat policy (map routing toward the boss, rewards, rest, shop, events, choices) |
| `rollout.py` | run-mode collector: N parallel full-run envs, combat-only transitions + GAE |
| `scenario.py` | scenario-mode collector: N parallel envs, each episode one isolated random fight + GAE |
| `ppo.py` | clipped PPO update (optax) |
| `train.py` | the training CLI (spawn envs → rollout → update → checkpoint → log) |
| `eval.py` | compare PPO vs. heuristic vs. random over seeds |
| `policies/jax_policy.py` | serve a checkpoint as `policy(state, options)` (JIT warm-up; declines out of combat) |

### Train

Two training modes share the same model, features, and PPO update:

**Run mode** (`--mode run`, the default) — full playthroughs, learning combat while the scripted
`navigator` drives non-combat; reward is HP retained + damage + kills + floor/win:

```sh
python -m lts2_agent.train --iterations 200 --envs 8 --steps 96 \
    --ckpt checkpoints/ppo --csv checkpoints/train.metrics.csv
```

**Scenario mode** (`--mode scenario`) — isolated random combats: each episode is one fight with a
**random character, a random 15-card deck from its pool, its starting relic + 5 random relics**, at
full HP, in a **random act-1/2/3 encounter** (weighted by `--elite-pct`/`--boss-pct`). Reward is the
fight outcome plus HP lost (`--sw-win`/`--sw-loss`/`--sw-hp`); the HP loss already adds back the
character's end-of-combat starter heal (e.g. Ironclad's Burning Blood +6) so it measures real combat
damage. This trains combat over a far wider spread of decks/relics/enemies than a normal run visits:

```sh
python -m lts2_agent.train --mode scenario --iterations 200 --envs 8 --steps 96 \
    --elite-pct 0.2 --boss-pct 0.05 --ckpt checkpoints/scenario \
    --csv checkpoints/scenario.metrics.csv
```

Each env runs its own `Lts2.AgentHost` process (one run per process — the game keeps state in
process-wide singletons), so `--envs N` is real parallelism. Metrics print to stderr and stream to the
CSV (run mode: floor/win/score; scenario mode: win-rate/HP-lost). Checkpoints (`<ckpt>` +
`<ckpt>.meta.json`) are written every `--save-every` iterations; `--resume` continues from one.
Omit `--character` for random-per-fight (the generalist); pass e.g. `--character Necrobinder` to
specialize.

**Deterministic eval set (scenario mode).** The per-iteration training win-rate is very noisy — random
fights, exploration, and some fights are simply unwinnable — so it's a poor progress signal. Every
`--eval-every` iterations the trainer instead plays a **fixed set of `--eval-seeds` seeded fights
greedily** (same seeds + same params → same fights), and logs `EVAL win / hpLost / hpFrac`. Because it
is deterministic, it moves only when the policy actually improves; **`hpFrac` (fraction of HP lost) is
the most sensitive flat-lining detector** since it keeps dropping even after win-rate saturates against
the unwinnable-fight ceiling. Eval columns are also written to the CSV.

### Evaluate

```sh
python -m lts2_agent.eval --policies ppo,heuristic,random --ckpt checkpoints/ppo --seeds 20
```

Reports win rate, mean/median/max floor, mean score, and mean combats survived per policy over the
same seeds — a learned policy should beat random and trend toward (then past) the heuristic.
Add `--mode scenario` to compare on isolated random fights (win-rate / HP-lost) instead of full runs.

### Inspect specific plays (closed evals)

To debug *why* the policy makes a given decision, `closed_eval.py` runs it on **fully-specified,
reproducible** situations — exact character + deck (so the hand is known) + encounter, and optionally
per-enemy HP for unambiguous spots like a free lethal — printing each option's features, the model's
scores, and the turn it plays:

```sh
python -m lts2_agent.closed_eval --ckpt checkpoints/scenario
```

The scenarios live in `closed_eval.py::SCENARIOS`; the underlying knob is
`Lts2Env.reset_combat(character=…, cards=[…], encounter=…, enemy_hp=[…])` (`CombatScenario.CreateExplicit`
on the C# side), usable directly to build your own.

### Serve in the TUI

**Recommended: a config file (no env vars per run).** The TUI auto-loads external agents from a
`lts2.agent.json` at the repo root (it walks up from its executable to find it). Copy the committed
template and you're done:

```sh
cp lts2.agent.example.json lts2.agent.json      # lists "PPO run (jax)" and "PPO scenario (jax)"
dotnet run --project src/Lts2.Tui
```

`lts2.agent.json` (gitignored — it holds machine-specific paths) lists one or more agents:

```json
{
  "agents": [
    {
      "name": "PPO (jax)",
      "command": "python/.venv/Scripts/python.exe",
      "arguments": "-m lts2_agent.decision_server lts2_agent.policies.jax_policy:policy",
      "workingDirectory": "python",
      "environment": { "LTS2_PPO_CKPT": "checkpoints/ppo" },
      "timeoutSeconds": 60
    }
  ]
}
```

Relative `command`/`workingDirectory` resolve against the config file's directory; `environment` vars
(e.g. the checkpoint path) are passed to the child. Point elsewhere with `LTS2_AGENT_CONFIG=/path/…`.

**Alternative: environment variables** (still supported, additive to the config file):

```powershell
$env:LTS2_PPO_CKPT = "checkpoints/ppo"
$env:LTS2_AGENT_CMD = "python/.venv/Scripts/python.exe"
$env:LTS2_AGENT_ARGS = "-m lts2_agent.decision_server lts2_agent.policies.jax_policy:policy"
$env:LTS2_AGENT_NAME = "PPO (jax)"
dotnet run --project src/Lts2.Tui
```

Either way, pick **"PPO run (jax)"** or **"PPO scenario (jax)"** from the Strategy menu; the `[tab]`
combat recommendation now comes from the net (it declines out of combat, so the game's own default
drives non-combat). The model is JIT-warmed at startup so the first recommendation stays under the C#
response timeout.

## Training dashboard

A local, offline web dashboard renders live and historical training charts straight from the
trainer's event files — no pip deps, no CDNs, no external fonts/scripts (stdlib `http.server` +
one self-contained `index.html` with hand-rolled SVG line charts). The trainer and the dashboard
share **only** the on-disk file contract, so you can watch any run — including ones started in the
background — with zero coordination:

- Runs live under a directory (default `checkpoints/runs/`), one subdir per run.
- `<run>/manifest.json` — `{runId, label, startedAt, kind, argv, config, gitSha, featureVersion,
  catalogSignature}`.
- `<run>/events.jsonl` — append-only, one JSON object per line:
  `{ts, phase, step, name, value, tags?}`. Outcome events carry `tags` such as
  `{act, room, character}` (fights) or `{act, room, character, mode}` (eval fights).

Launch it:

```sh
python -m lts2_agent.dashboard --dir checkpoints/runs --port 8777   # also: --host (default 127.0.0.1)
```

Then open `http://127.0.0.1:8777`. The UI has a run sidebar (multi-select checkboxes to overlay
runs; a green dot marks runs whose last event is < 10s old), a metric / group-by / bucket toolbar
with a poll pause button (auto-refresh every 2s), and preset breakdown buttons — **Win by room**,
**Win by act**, **HP lost by room**, **Eval greedy vs sampled win**. Every series shows its total
sample count `n` in the legend and per-point in the tooltip, so a rate over few fights reads as
thin as it is.

HTTP API (all JSON): `GET /api/runs` (newest-first summaries), `GET /api/runs/<id>/meta`
(metric names, tag keys, maxStep), `GET /api/runs/<id>/series?name=&group_by=<tagKey|none>&bucket=<int|auto>`
(per-group downsampled points, each `{step, value=mean, n=count}`). Event files are tailed
incrementally (only appended bytes are re-read) and a truncated final line mid-write is tolerated.

To try it without a real run, generate synthetic data (optionally live-appending):

```sh
python -m lts2_agent.dashboard.demo --dir checkpoints/runs --live
```

## Oracle prober (replay-based ground truth)

The world-model's *predictor* will predict the next observation given a state and an action. To score
it we need **ground truth**: for a fixed set of combat *positions*, the true next observation for
**every** legal action. The emulator is deterministic — same seed + same reset params + same action
sequence reproduces a fight exactly — but there are no mid-combat snapshots, so a position is reached
by **replaying an action prefix** from a seeded combat start. `lts2_agent.oracle` does exactly that:

1. a **probe** freezes a reproducible position: `{probeId, resetParams (seed/character/elitePct/
   bossPct/starterDeck/act), actionPrefix (option indices), meta}` where `meta` captures
   `{act, roomType, character, turn, phase, optionCount}` for stratification;
2. the **oracle runner** replays each probe, enumerates its legal options, and for each option index
   replays `reset + prefix + [i]` to record the resulting full observation — the ground-truth next
   state for that action.

**Evaluation-only, by construction.** Every probe seed lives in a reserved namespace prefixed with
**`PROBE-`**. **Training collectors must never use that prefix** — it keeps probe fights structurally
excluded from any training corpus. `oracle.validate_probe_seed` enforces the rule (the builder always
prefixes) and `oracle.assert_not_probe_seed` is the guard collectors call.

Three commands (all against the built `Lts2.AgentHost`):

```sh
# 1. Freeze a probe set (reproducible from --master-seed). Spans acts 0-2, a monster/elite/boss room
#    mix (via the elite/boss pct knobs), all characters, and varied replay depth (0-15 steps).
python -m lts2_agent.oracle build --n 300 --out lts2_agent/data/probes.json

# 2. Replay -> ground-truth next states; writes a gzip-JSONL shard, one record per probe.
python -m lts2_agent.oracle run --probes lts2_agent/data/probes.json --out shard.jsonl.gz --envs 4

# 3. Determinism spot-check (also available as `run --verify`): double-replay each position, assert
#    the serialized state is byte-identical.
python -m lts2_agent.oracle verify --probes lts2_agent/data/probes.json --sample 40
```

- **Reproducibility gate.** ~5% of deep-replay fights are genuinely non-reproducible (an RNG not
  reseeded per fight, or async-combat-pump ordering) — either fully chaotic or *flaky* (occasional
  divergence). At build time every candidate is replayed `--reproduce-checks` (default 8) times on a
  **second, independent host process** and kept only if every replay is byte-identical, so every
  committed probe is a stable, cross-process ground-truth position.
- **Shard record schema** (gzip JSONL, one line per probe):
  `{probeId, position: {state, options, done, info}, results: [{action: i, obs: <next-obs>} | {action: i, error: "..."}], meta}`.
  A probe whose position replay itself fails is recorded as `{probeId, error}` and skipped. Per-action
  and per-probe env errors are tolerated (recorded, never fatal). Progress streams to **stderr**.
- **Parallelism.** `--envs N` runs probes across N host processes (a thread pool over env instances,
  like the trainers — env I/O releases the GIL).

The committed `lts2_agent/data/probes.json` is a small 40-probe set (kept light on purpose); the full
few-hundred-probe evaluation set is built at CP2.

## Transition corpus (supervised world-model data)

The world model (encoder/predictor) is trained **supervised** on logged transitions. `lts2_agent.collect`
plays scenario combats with cheap policies and logs **every** decision point — combat moves *and*
mid-combat `Choice` selections — as one `(state, action, next-state)` record. `lts2_agent.corpus` is the
store; `lts2_agent.corpus_report` is the CP2 composition report. Stdlib only.

```sh
# Collect (N parallel host processes; broad+realistic regimes, random/heuristic policies mixable).
python -m lts2_agent.collect --envs 8 --fights 500 --regime mixed --policy mixed \
    --out python/data/corpus --run-label demo
#   --transitions N instead of --fights; --regime broad|realistic|mixed; --policy random|heuristic|mixed;
#   --character (default random) --act -1(any)/0/1/2 --elite-pct/--boss-pct.

# CP2 report: composition, realistic-deck distributions, a 20-deck sample, determinism spot-check.
python -m lts2_agent.corpus_report --corpus python/data/corpus        # add --json for machine-readable
```

- **Layout** — sharded gzip JSONL under the corpus root: `<root>/{train,val,test}/<run-label>-<NNNNN>.jsonl.gz`,
  one JSON record per line, shards capped at ~2000 records. The default root is `python/data/corpus/`
  (gitignored — the shards are large and regenerable). A whole fight is written **atomically** at its end;
  a fight that errors (the env throws sporadically) or exceeds the ~90-decision cap is dropped cleanly.
- **Record schema** (contract 4, lossless raw wire observations):
  `{seed, scenarioMeta, t, state, options, actionTaken, nextState, nextOptions, rewardComponents, done, info}`.
  `scenarioMeta = {deckSpec, removedCards, addedCards, act, room, character, encounter, policy, regime}`;
  `actionTaken` is an option index (or a `cardIndices` list); `rewardComponents` are raw before/after
  scalars (player currentHp/block, summed enemy HP) — **no reward function applied**, the trainer derives
  its own.
- **Split rule** (leak-proof, deterministic): the split is a pure function of the **fight seed** —
  `crc32(seed) % 100` → 0-89 train / 90-94 val / 95-99 test (`corpus.split_for_seed`, used by writer and
  reader alike). All records of a fight share the seed, so a fight lands wholly in one split; the same
  seed always maps to the same split. Train/val/test leakage is structurally impossible.
- **Seed namespaces**: collector fight seeds are `CORPUS-<run-label>-<env>-<counter>`. The `PROBE-`
  namespace (oracle eval set) is refused, and records from an `explicit` deckSpec (closed-eval fixed
  instances) are refused — training data is always sampled distributions, never fixed instances.
- **Dashboard**: a collection run appears like a training run (metrics `kind="collect"`) — aggregate
  `collect.transitions_total/fights_total/errors_total/transitions_per_s` plus per-fight `fight.won`/
  `fight.hp_lost` tagged `{act, room, character, regime, policy}`.

## Tokenizer — the world-model parity contract

`lts2_agent.tokens` is the **successor to `features.py`** for the world-model stack (design §4.1). Where
`features.py` hand-crafts a fixed scalar vector — the "feature treadmill" where every unmodelled mechanic
is invisible until someone adds a feature and bumps `FEATURE_VERSION` — the tokenizer encodes a state as a
**set of typed entity tokens**. Rule: *if the wire exposes it, tokenize it.* New mechanics arrive as new
catalog ids + generic numeric fields, never as bespoke features. `TOKENIZER_VERSION` (+ the four catalog
signatures) is the new train/serve parity stamp, playing the role `FEATURE_VERSION` plays for the PPO
model. NumPy-only (no torch/jax), so it imports everywhere. The PPO baseline still uses `features.py`.

```sh
# CP3 artifact: coverage + round-trip report over the corpus (every state AND nextState).
python -m lts2_agent.tokens --check python/data/corpus            # add --limit N to sample
# Exits nonzero on any lost field or any round-trip mismatch.
```

- **Token types** (each token carries a token-type id): a **global** token (phase/side/turn-phase +
  act/floor/ascension/score/energy/stars/turn/gold/…); one **card** population row per distinct card
  CONTENT across hand/draw/discard/exhaust + the offered cards of a pending choice (card-catalog index +
  the live `CardView` dynamic fields: cost/costsX/starCost/damage/baseDamage/block/baseBlock/summon/
  upgraded/canPlay/replayCount/enchant/affliction + a hashed multi-hot of `addedKeywords`), carrying a
  **per-zone count vector** `count_{hand,draw,discard,exhaust,offered}` instead of a zone id — see the v3
  note below); one **creature** token per
  player/Osty/enemy (hp/maxHp/block/active + identity); **power** tokens (power-catalog index + amount,
  parented to a creature); **intent** tokens (type/damage/hits, parented to an enemy); **orb**, **relic**
  (relic-catalog index), **potion** (potion-catalog index, per belt slot incl. empty), and a **pending-
  choice** token (min/max select + upgrade flag). Fixed-shape padded arrays + boolean masks batch cleanly.
- **The draw pile (and every card zone) is an unordered MULTISET.** Card population rows are sorted by
  their full content tuple, so the wire's shuffle order can *never* leak. Two shuffles of the same pile
  produce byte-identical tokens (`test_tokens.test_draw_pile_is_unordered_multiset`).
- **`TOKENIZER_VERSION = 3` — factored population rows (the T3 "expert-per-category" data layer).** Zone
  leaves the card grouping key: there is **one row per distinct card CONTENT** (catalog id + every live
  dynamic field + keywords), carrying a **per-zone count vector** `count_{hand,draw,discard,exhaust,
  offered}` (symlog, the trailing five `CARD_NUM` columns) in place of a zone id. A content that occupies
  several piles is a single row whose counts spread across zones; a card whose live fields differ across
  zones (a cost-reduced copy in hand vs its full-cost twin in draw) stays a *separate* row — divergence is
  correct. Rationale: population membership becomes **structural** — a card moving hand→discard is the same
  row with the count shifting between two columns, so the future predictor expresses zone transitions as
  count arithmetic and creation/transform as rows appearing/disappearing. `detokenize` expands the count
  vector back into per-instance-per-zone dicts, so the **canonical dict is byte-identical to v1/v2's** and
  every canonical-dict consumer (`statefmt`, `legal_actions`, the report card) is untouched. Measured
  (shard-strided 336k-state scan of `data/corpus`): mean 14.2 instances → 10.2 rows (1.39× shorter), v3
  rows worst case 32 (v2 zone-scoped grouped max 42, v1 instance max 82); `MAX_CARDS` stays 64 (generous
  slack). Round-trip/coverage contract unchanged: 0 lost fields, 0 mismatches. (v2 was count-grouped
  tokens *with* zone in the key; v1 was raw per-instance tokens.)
- **Per-field integer ranges — the v3 exactness contract.** Every numeric column carries a measured
  `(lo, hi, resolution)` range in `wm/spec.py` (`NUMERIC_RANGES`), scanned from the corpus by `python -m
  lts2_agent.wm.ranges` (footprint's streaming pattern) — e.g. energy 0..40, creature HP 0..1000 (the
  `999999999` sentinel caps loud), gold 0..5000, per-zone counts 0..40. These are the exact per-field
  domains a **future per-field decoder** bins against (`(hi−lo)//resolution+1` bins) instead of regressing
  one shared symlog float. The tokenizer itself still **stores symlog** (cache/decoder compat), so the
  integer round-trip stays exact via `round(symexp)` inside the ±`NUM_CLIP` clamp; the ranges are the
  *decode* contract, and `spec.clamp_to_range` clamps any out-of-`[lo,hi]` value **loudly** (the documented
  out-of-distribution signal).
- **Numerics use symlog** (`sign(x)·log1p|x|`, DreamerV3-style) with a ±`NUM_CLIP` (1e5) clamp — bounded
  (no encoder blow-ups) and **exactly invertible** for integer game quantities, which is what makes the
  round-trip validator exact. The clamp only saturates the game's `999999999` "no maximum" select sentinel
  and any pathological scaling outlier. Every categorical is a **catalog index** (cards/powers/relics/
  potions — exact inverse) or a small **fixed enum** (token/intent/target/card types, phases — enumerated
  from `GameState.cs`).
- **Catalogs** (`lts2_agent.catalog`, generalizing `card_catalog`): each of cards/powers/relics/potions
  gets a stable dense id→index (0 = none/unknown), a static multi-hot table (categorical one-hots ++ flags
  ++ tags/keywords/var-keys), and a content signature. Null-tolerant: on a fresh clone with no dump, falls
  back to CRC32 hashing into a fixed vocab. Regenerate the dumps with
  `Lts2.AgentHost --dump-{cards,powers,relics,potions} > python/lts2_agent/data/<kind>.json`.
- **Coverage contract**: `coverage_check(state)` walks the raw wire dict and classifies **every** field as
  covered / waived / lost; `lost` must stay empty over the corpus. Waivers (in `tokens.WAIVERS`, each with
  a reason): non-combat room views (`map`/`rewards`/`bundleChoice`/`event`/`shop`/`restSite`/`treasure`/
  `crystalSphere` — the tokenizer is a *combat* world-model), `seed`/`netId` (identifiers), `deck` (the
  persistent run deck; in combat the live cards are the four piles), and per-card `poolId` (static, already
  in the card-catalog row). A handful of open string ids with no catalog dump (monster/character/orb/
  enchant/affliction ids, granted keywords) are **covered-lossy** — hashed into fixed vocabs (`LOSSY_FIELDS`
  documents each); they are tokenized but do not round-trip back to a string.

### PPO-on-tokens (the model-free representation upgrade, roadmap 2.2)

`lts2_agent.model_tokens` is a small **set-transformer actor-critic** over the tokenizer, trained under the
*existing* PPO algorithm — the design's "Alternative A" upgrade (§6.A), banked before the world-model
depends on the tokenizer. Per-token-type embedders (the **card** embedder shared between state card tokens
and legal-option cards) project into a shared `d_model` (default 160, ~1.3M params); creatures fold in
their powers/intents (scatter-add) then self-attend; learned latent queries attention-pool the whole token
set into a state context `z`. Each legal option is scored as **(kind ⊕ option-card/potion embedding ⊕ the
target creature's embedding, gathered by `targetCombatId → creature slot` ⊕ `z`)** through a masked softmax
(exactly like `model_torch`), with a tanh-bounded ±20 value head. Checkpoints are stamped with
`tokenizer_signature()` and reject a mismatch loudly, the way `model_torch` rejects a `FEATURE_VERSION`
mismatch.

The rollout (`rollout_torch`) and PPO update (`ppo_torch`) are shared with the features baseline via a small
`adapters.py` seam (a defaulted `adapter` parameter) — so `train_torch` is byte-for-byte unchanged and stays
the recorded baseline. The trainer is a separate CLI (`train_tokens`), same `ScenarioConfig` knobs, reward,
and metrics stream (as a `kind="ppo-tokens"` run so the dashboard overlays it on the baseline):

```sh
# Full baseline-comparison run (matches the M0.5 baseline scenario settings).
python -m lts2_agent.train_tokens --envs 16 --iterations 300 --eval-every 10 \
    --ckpt checkpoints/tokens_m2.pt --run-label tokens-ppo
```

Serve a trained token checkpoint into the TUI with `lts2_agent.policies.torch_tokens_policy` (sampled by
default — argmax collapses onto EndTurn, same as the features policy; `LTS2_PPO_TOKENS_CKPT` sets the path).

## World-model encoder/decoder (roadmap 3.1, design §4.2–4.3)

`lts2_agent.wm` is the first world-model module: an **encoder** that compresses a tokenized combat state
into a normalized latent `z`, and a **symbolic decoder** that reconstructs the structured state from `z`
alone. The decoder is the training signal, the anti-collapse anchor (design §4.3 — reconstruction makes
the JEPA collapse mode impossible), and — later — the debugger. It trains **supervised** on the transition
corpus (no reward, no env): every record's `state` and `nextState` (~2M states) autoencoded.

- **Encoder** (`wm/encoder.py`): per-token-type input projections into `d_model` (default 256; cards/
  powers/relics/potions also gather their static-catalog row), a token-type embedding, `enc_layers`
  (default 4) of pre-norm self-attention over the packed token set (key-padding-masked — no manual
  sort/pack needed at these batch sizes), then Perceiver-style **attention pooling** (learned latent
  queries) into a latent vector (`z_dim`, default 512).
- **SimNorm latent** (design §11 delta): `z` is split into groups of `simnorm_group` (default 8) and
  softmax'd within each group (TD-MPC2's SimNorm), so the latent is a concatenation of probability
  simplices — bounded (each group sums to 1), so it can neither explode nor collapse to a constant scale.
  Chosen over plain L2 because grouped-simplex latents are the published default for latent world models
  and preserve more categorical structure at equal width.
- **Decoder** (`wm/decoder.py`): `z` → a few memory tokens; per token type a bank of **learned slot
  queries** cross-attends into the memory and self-attends (`TransformerDecoderLayer`), then per-type
  heads emit the tokenizer's array space directly — categorical logits per `*_idx` column (cross-entropy),
  a numeric vector per `*_num` block (**MSE on the symlog values** — the same array `tokens.detokenize`
  inverts; symlog already compresses and integer quantities round-trip exactly, so two-hot buys nothing
  here), a per-slot **presence** logit for each variable-length type (BCE), and card-keyword multi-hot
  (BCE). Canonical-dict reconstruction reuses `tokens.detokenize` verbatim (never reimplemented).
- **Field spec** (`wm/spec.py`) is the single description of the array layout both the encoder embedders
  and decoder heads iterate, so the model's output space *is* the tokenizer's array space; vocab sizes come
  from the live catalogs/enums. Checkpoints stamp `tokenizer_signature()` and reject a mismatch loudly.

**Latent shape A/B (`--latent-mode`, design §10 / CP4 decision).** The latent structure between the pool
and the decoder is a switch, so a same-budget comparison isolates *only* it (encoder stack, per-type heads,
losses, data, metrics are all unchanged):

- `flat` (default) — the Perceiver pool's latents are flattened and projected to a single SimNorm vector
  `z` (`z_dim`, default 512), which the decoder re-expands into `n_mem` memory tokens. This is the original
  path, **byte-identical when selected** (existing checkpoints load unchanged).
- `tokens` — the pool keeps `--latent-k` latent tokens (default 16, `d_model` each) *as* the latent: no
  flatten, no `z_dim` projection, SimNorm applied **per latent token** (each token is its own concatenation
  of simplices), and the decoder consumes those tokens **directly as its memory** (dropping the `z →
  memory` expansion). This removes the flatten-to-512 squeeze — the suspected constraint on card-identity
  reconstruction over big multisets. Dropping the two projections makes `tokens` the *smaller* model
  (~7.0M vs ~10.1M params at the defaults; the ~3.15M delta is exactly `encoder.to_z` + `decoder.to_mem`).

Pick `flat` unless you are running the A/B; the checkpoint meta stamps `latent_mode`/`latent_k` and load
rejects a mode mismatch loudly (so `--resume` cannot cross a flat/tokens boundary). Which mode wins the
CP4 latent-shape decision is settled by reconstruction quality at equal training budget.

**Loss/recipe probe flags (3.1 experiment series).** Three independent switches feed the one-change-at-a-
time 5k-step probes (each pairs with `--halt-step` to share the long cosine schedule); every one defaults
OFF and is byte-identical to the tokens control when off. **`--num-head twohot`** (default `mse`) replaces
the numeric regression heads with DreamerV3-style two-hot classification: each numeric column is predicted
over a fixed 64-bin grid spanning the symlog clamp range `[-symlog(NUM_CLIP), +symlog(NUM_CLIP)]`, trained
with cross-entropy against the two-hot (linear-interpolation) target and decoded as the softmax expectation
over bins — so `reconstruct_arrays`/report consume the decoded symlog value exactly as before; the recipe is
stamped in the checkpoint meta and `--resume` rejects an mse/twohot mismatch. **`--card-ce balanced`**
(default `plain`) class-frequency-balances the cross-entropy of the card-identity column *only* (card
categorical column 0): each class is weighted `1/sqrt(freq)` from the corpus card-index distribution
(scanned once from `--card-ce-states` train states, cached to `card_ce_w_<sig>.npy` keyed by tokenizer
signature so restarts are cheap), normalized so the frequency-weighted mean weight is 1 (loss scale
unchanged); all other columns are untouched. **`--ema DECAY`** (default `0` = off, e.g. `0.999`) maintains
an exponential moving average of the weights updated every step; **val passes evaluate the EMA weights**
(swapped in and back out around each pass) and checkpoints save the EMA shadow in a `.pt.ema` sidecar
(`ema_decay` in meta) that `--resume` restores alongside the raw state. The flags compose (e.g.
`--num-head twohot --ema 0.999`).

**Relic decode (CP4 fix).** The relic set is the decoder's worst structural field: 24 independent per-slot
categoricals over the ~298-relic catalog, so under uncertainty several slots argmax onto the same corpus-
common relic — a duplicate that no real run can hold (and a rare relic gets dropped). Two independent fixes
address this. (1) A **decode-time dedup** (`reconstruct_arrays(dedup=True)`, `eval_encdec --dedup`) reassigns
present relic slots greedily by confidence: order slots by their max softmax probability, and give each its
highest-probability id among those not yet claimed. It is pure inference — no training effect, applies to any
existing slot-head checkpoint — and lifts the gate checkpoint's `relic_set_f1` from **0.920 → 0.995** over
2000 val states (`state_dist`/`action_snr` unchanged: 0.0291→0.0289, 5.86→5.89). (2) **`--relic-head set`**
(default `slots` = the per-slot head, byte-identical) replaces the relic branch with ONE multi-hot head over
the catalog: `logits [B, relic_vocab]`, trained with BCE against the multi-hot of present relic ids (folded
into the categorical-loss bucket, replacing both the relic identity-CE and relic presence terms), and decoded
as top-`k` for `k = clamp(round(Σ sigmoid), 0, MAX_RELICS)` — duplicate-free by construction. Stamped in the
checkpoint meta; `--resume` rejects a slots/set mismatch. It composes with `--num-head`/`--card-ce`/`--ema`.
Limitation: a top-`k` set cannot represent a genuine *duplicate* relic, and ~0.67 % of corpus states do hold
one (e.g. `INFUSED_CORE`, `TOUCH_OF_OROBAS`, `BLACK_BLOOD`); those states lose the repeat.

```sh
# Train (streams the train split; per-field reconstruction metrics stream live to the dashboard).
python -m lts2_agent.train_encdec --steps 50000 --batch 384 --val-every 500 \
    --ckpt checkpoints/wm_encdec.pt --run-label wm-encdec        # --resume to continue

# Token-set latent variant (same budget/knobs, different latent structure).
python -m lts2_agent.train_encdec --latent-mode tokens --latent-k 16 --steps 50000 --batch 384 \
    --ckpt checkpoints/wm_encdec_tokens.pt --run-label wm-encdec-tokens

# CP4 artifact: full-split report card (overall + by-act) against a checkpoint.
python -m lts2_agent.eval_encdec --ckpt checkpoints/wm_encdec.pt --split val   # --json for machine form
```

**Pre-tokenized cache (one-time, for speed).** On-the-fly Python tokenization can bottleneck the trainer
on a corpus that never changes between runs (worse still when the GPU is shared with another job). Build a
pre-tokenized cache once so tokenization leaves the critical path entirely:
`python -m lts2_agent.wm.cache build --corpus data/corpus --out data/corpus_tok --workers 8` streams every
record per split, tokenizes with a multiprocessing pool, and writes compressed `.npz` array shards + a
`manifest.json` stamped with the tokenizer signature (it auto-verifies ~200 states against a fresh
tokenize). The 2.0M-state corpus builds to ~179 MB in ~39 min (~800 states/s, parent-side gzip/JSON
bound). `train_encdec` uses the cache automatically when `data/corpus_tok` exists and its signature matches
(mismatch = loud error, not silent fallback; pass `--cache ""` to force on-the-fly). The cache stores
**both** each record's `state` and `nextState` (no dedup) to preserve exact training-distribution parity
with the live loader. Measured on an RTX 3090: the cache data path delivers **~7400 states/s** (vs ~960
states/s single-thread on-the-fly), so training is now fully GPU-bound — the encoder/decoder
forward+backward itself caps this box at **~470 states/s** (fp32, no flash-attention on the Windows torch
build), i.e. ~11-12 h for a 50k×384 run. **Rebuild the cache whenever the corpus changes or the
tokenizer/catalog signature bumps** (`TOKENIZER_VERSION` or any catalog); it is gitignored
(`python/data/corpus_tok/`). Build into a **new** dir on a version bump rather than overwriting the old
one — e.g. **tokenizer v3** (factored population rows) needs
`python -m lts2_agent.wm.cache build --corpus data/corpus2 --out data/corpus_tok_v3 --workers 8`
(then `train_encdec --cache data/corpus_tok_v3`). The old v1/v2 caches stay intact and reject
loudly if a v3 run points at one.

Metrics land as a `kind="wm-encdec"` run under `checkpoints/runs/`. **Per train step-window**
(phase=`train`): `train.loss`, `train.loss_categorical`, `train.loss_numeric`, `train.loss_presence`,
`train.lr`, `train.states_per_s`. **Per val pass** (phase=`eval`) the per-field report card:
`eval.card_id_top1`, `eval.card_zone_acc`, `eval.power_id_top1`, `eval.power_amount_mae`,
`eval.creature_hp_mae`, `eval.creature_block_mae`, `eval.intent_damage_mae`, `eval.energy_acc`,
`eval.relic_set_f1`, `eval.potion_set_f1`, `eval.hand_size_acc`, `eval.pile_size_acc`,
`eval.pending_choice_acc`, and the aggregate `eval.exact_state_rate` (fraction of val states whose full
decoded canonical dict equals the original after detokenize-level quantization). MAEs are RAW game units
(symexp'd), not symlog space. Each eval metric is emitted a second time tagged `{"act": …}` so the
dashboard's group-by works. Note: per-slot metrics (card-id, power-id) are measured **slot-aligned** to the
tokenizer's content-sorted target order; because that order is a deterministic function of content the
decoder can learn it, but the slot-assignment ambiguity makes these a conservative floor — `exact_state_rate`
(set-based, order-invariant via `detokenize`) is the honest aggregate.

`eval.action_snr` = `report.ACTION_FOOTPRINT` / `state_dist` — how many median-actions' worth of change
the decoder can still resolve. The footprint is measured in `state_dist` token-field units, which the
tokenizer's field universe defines, so it is **re-measured per tokenizer version** (cross-tokenizer
comparisons use `action_snr`, not raw `state_dist`): `python -m lts2_agent.wm.footprint --corpus
data/corpus --n 3000` prints per-kind medians and the constant to set. Current `ACTION_FOOTPRINT = 0.1224`
(tokenizer v3; v2 was 0.1704, v1 0.1303 — v3's per-zone count vector means a PlayCard usually just shifts
counts between two columns of one shared row instead of moving a whole card token, so its footprint fell
sharply: PlayCard median 0.050).

### Factored expert autoencoder (`--arch factored`, roadmap M3.5)

`--arch factored` swaps the single monolithic latent for the **T3 expert-per-category** design the v3
tokenizer was built for (`wm/experts.py`, `wm/model_factored.py`). Each entity **category** is its own
independent set autoencoder with its own **latent slice**; the state latent is the concatenation of the
slices, a **named, offset-addressable layout** (`scalars · creatures · cards · relics · potions · orbs`,
stamped in the checkpoint meta) the M4 predictor will read/write by slice. There is **no cross-category
attention inside the AE** — independence is deliberate (a card expert never attends to creatures), which
both shrinks each attention scope and gives the predictor a clean per-slice seam; cross-category coupling
is the predictor's job. `--arch mono` (default) is the unchanged monolith.

Three tiers:

- **Tier 1 — scalar codec (parameter-free, exact by construction).** The global token (3 enum
  categoricals + 14 numerics) and pending (4 numerics) encode to a deterministic slice: one-hot for the
  small enums, a fixed **binary code of the `NUMERIC_RANGES` bin index** for each numeric. Encode *and*
  decode are fixed functions with no learned weights, so the round-trip is **exact for any in-range
  integer regardless of training** — `eval.scalar_exact` reads 1.0 from the very first val pass (a wiring
  canary). The only misses on real data are the documented loud clamps (the `999999999` no-limit
  `maxSelect`/HP sentinels): 0.9999 on corpus2 val is 26/24000 choice states hitting the `maxSelect`
  cap, exactly the intended out-of-distribution signal.
- **Tier 2 — small experts.** creatures (folds its powers + intents into one set expert, keeping the
  parent-slot association), relics (a multi-hot **set-membership** head, duplicate-free by construction —
  the monolith's `--relic-head set` ported over), potions (per-slot categorical, since potions can
  duplicate), orbs. The small single-type experts run at 1 encoder / 1 decoder layer.
- **Tier 3 — card-population expert.** the largest slice, a set enc/dec over the v3 population rows
  (content categoricals + keyword multi-hot + dynamic numerics + the per-zone count vector).

All learned numerics decode through **per-field range-bin classification** (`RangeBinHeads`) rather than
the monolith's shared symlog MSE: creature HP gets resolution-1 bins over `[0,1000]`, so an
in-distribution value decodes to the *exact* integer (no ±1 rounding tail). Each head still emits the
identical symlog `num` block (argmax bin → integer → symlog), so `reconstruct_arrays`/`report` consume
factored outputs unchanged. (Each expert's encoder carries an always-valid sentinel token so an empty
category — no orbs/potions in a state — never yields a fully-padded-attention NaN.)

**New metrics** (factored only): **`eval.expert_dist`** — each expert's share of `state_dist`, emitted
once per expert tagged `{"expert": …}`, partitioning `state_dist` *exactly* (the dashboard groups by
`expert` to overlay the per-decoder curves); **`eval.scalar_exact`** — the tier-1 canary. Every existing
report-card metric flows unchanged.

```sh
# Factored AE (defaults: latent_dim 3188, ~22.9M params — cards the largest slice). Same knobs/cache as mono.
python -m lts2_agent.train_encdec --arch factored --cache data/corpus_tok_v3 --corpus data/corpus2 \
    --steps 50000 --batch 384 --val-every 500 --ckpt checkpoints/wm_factored.pt --run-label wm-factored
```

Checkpoints stamp `arch=factored` + the slice layout; a load rejects a non-factored or layout-mismatched
checkpoint. Measured on an RTX 3090 (450×384, bf16, v3 cache): losses fall smoothly, `scalar_exact`≈1.0
and `energy_acc`=1.0 from the first val, and it runs **~12 % faster than the monolith** (~1.69 k vs
~1.50 k states/s) — the smaller per-expert attention scopes outweighing the extra per-expert kernels.

#### Per-expert training: train → keep → compose (roadmap M3.5)

Because the experts are **parameter-disjoint** (no cross-category attention), they can be trained
**independently and sequentially** — the product-owner strategy: train one expert until its slice hits a
high exactness bar, keep it, move to the next, and never retrain a healthy expert because another is
struggling. Three CLI seams make this a workflow:

- **`--train-experts a,b,…`** (factored only; default = all) trains *only* the named experts. The rest are
  **frozen** — excluded from the optimizer (so they stay byte-identical) *and* their encode/decode is
  **skipped entirely** during the step, so a solo small-expert run pays only that expert's compute. A solo
  relic run (~3.5 M trainable params of the 22.9 M model) hits a much higher `states/s` than the joint run.
- **`--val-experts trained-only`** (default `all`) restricts the val pass to the trained experts,
  reconstructing only their token types — so a solo run doesn't pay the full-model report-card decode. It
  emits just that expert's `eval.expert_dist` / `eval.expert_exact` (+ `eval.relic_set_f1` for relics),
  and the `.best` sidecar is driven by the trained experts' mean `expert_dist`.
- **`--init-expert-from name=ckpt …`** warm-starts one expert's weights from the matching slice of an
  existing full factored checkpoint (e.g. `cards=checkpoints/wm_t3_v3.pt.best`), so a solo run can seed
  from a joint run's partial progress. It validates the source slice width + build config match.

**`eval.expert_exact`** (new, factored only; tagged `{"expert": …}` like `expert_dist`) is each expert's
**"done" bar**: the fraction of val states whose slice-owned token types reconstruct *exactly* (array
space, integer-rounded, presence included). Train an expert until its `expert_exact` plateaus, then keep it.

**Solo-run dynamics (roadmap M3.5).** Three knobs fix the slow, spiky, near-linear solo learning curves
(the log-shaped early ramp was missing; e.g. orbs barely moved over 12 k steps despite being a far simpler
problem than cards):

- **`--num-targets twohot|hard`** (factored; default **`twohot`**) sets the range-bin numeric target
  geometry. `hard` is one-hot CE on the exact integer bin — a near-miss (bin *k*±1) costs exactly as much
  as bin *k*±100, so the head never learns the numbers' **ordinal** structure. `twohot` is a distance-aware
  symmetric triangular target centred on the true bin (`{0.25, 0.50, 0.25}` over the three central bins),
  so nearby-bin predictions get partial credit — restoring the metric structure the monolith's two-hot
  gave. (The factored heads bin at *integer* resolution, so a literal two-adjacent-bin split would degrade
  to one-hot; the small symmetric kernel is the faithful generalization for a fine grid.) **Decode stays
  argmax** — it preserves the exact-bin contract, whereas an expectation over a soft prediction can straddle
  and round to a neighbour (measured: argmax's exact-bin rate ≫ expectation's).
- **`--focus-present R`** (factored **solo** runs only) oversamples states where a trained expert has ≥1
  present token: a fraction `R` of each batch is drawn from present states, `1−R` from empty ones (kept so
  the presence head stays calibrated). A sparse expert (orbs is present in only ~13 % of states) otherwise
  spends ~87 % of every batch on empty slots, starving its id/numeric heads. The sampler bounds both pools
  (present + empty), so a dense expert (creatures/potions, ~always present) is a cheap no-op. Needs the
  pre-tokenized cache.
- **Batch scaling.** Solo runs are tiny in VRAM (a single small expert) and the cache streams ~12 k
  states/s, so the batch-384 default gives needlessly noisy gradients. `--batch` is respected as-is; the
  probes below use **1536** for the small experts (well under the 3090's VRAM), with LR scaled ~√(batch)
  (`3e-4 → 6e-4` at 4× batch — √-scaling is steadier than linear at this ratio).

**Overfit-one-batch gate** — `python -m lts2_agent.wm.overfit --expert all` trains each expert alone on a
single fixed (present-heavy) batch and reports the step at which its `expert_dist` drops below `0.01`.
Memorizing one batch is the easiest possible task, so an expert that *can't* has a **wiring bug** — this is
the primary correctness gate for the solo path. It also prints each numeric type's exact-bin rate under the
two decodes (argmax / expectation), the evidence for the argmax decode choice. `--synthetic` runs it with
a random in-range batch (no corpus/GPU) — the CLI smoke path.

Once each expert is trained, **compose** the kept slices into one standard full factored checkpoint (the
artifact the M4 predictor consumes):

```sh
# Solo-train relics from scratch, focused val, on the GPU (tiny — batch 512 is fine).
python -m lts2_agent.train_encdec --arch factored --train-experts relics --val-experts trained-only \
    --cache data/corpus_tok_v3 --corpus data/corpus2 --steps 6000 --batch 512 --val-every 1000 \
    --ckpt checkpoints/relic_solo.pt --run-label relic-solo

# Warm-start a cards solo run from the joint run's cards slice.
python -m lts2_agent.train_encdec --arch factored --train-experts cards --val-experts trained-only \
    --init-expert-from cards=checkpoints/wm_t3_v3.pt.best --cache data/corpus_tok_v3 \
    --corpus data/corpus2 --steps 20000 --batch 384 --ckpt checkpoints/cards_solo.pt

# Compose: assemble a full checkpoint from the kept per-expert runs (--base fills the rest).
python -m lts2_agent.wm.compose --out checkpoints/wm_composite.pt \
    --base checkpoints/wm_t3_v3.pt.best \
    --experts relics=checkpoints/relic_solo.pt.best cards=checkpoints/cards_solo.pt.best

# eval_encdec loads a factored/composite checkpoint transparently (auto-detects arch from meta).
python -m lts2_agent.eval_encdec --ckpt checkpoints/wm_composite.pt --split val --cache data/corpus_tok_v3
```

Compose is strict: every source must be factored + tokenizer-compatible, all sources must agree on the
shared global config, and each pulled expert's slice width + build config must match the composite (a
slice can only land in a checkpoint it fits). The composite's per-expert provenance is recorded in the
checkpoint meta (`composed_from`). Each factored checkpoint's meta also carries a per-expert **stamp**
(`experts`: slice layout + tokenizer signature + build kwargs), the contract compose/warm-start validate
against; the expert weights themselves live in the one full `state_dict` under `experts.<name>.*`.

**Relic decode variant (bake-off).** The relic expert supports two decode heads via **`--fac-relic-head`**
(factored only): `set` (default) is the multi-hot set-membership head + count head (duplicate-free by
construction); `slots` is the monolith-style per-slot categoricals + inference **dedup** (greedy-by-
confidence unique assignment — `_dedup_slot_ids`, auto-enabled in val/eval for a slots model).
**`--relic-dec-layers N`** (default 1) deepens the relic decoder for a set-head capacity probe. These feed
the M3.5 relic bake-off (the set head only ships if it *beats* slots+dedup on `expert_exact`).

## Decoded-state printer + diff (roadmap 3.2)

`lts2_agent.statefmt` is the human window onto a **canonical state dict** — the exact shape
`tokens.detokenize` produces, whether from `detokenize(tokenize(raw wire))` or from a decoder's output.

- `format_state(cv, hash_names=None)` renders it compactly: player / Osty / enemies with
  hp/block/powers/intents, energy/stars/turn, the hand with per-card cost/dmg/block/upgrade,
  draw/discard/exhaust as counted multisets, relics, potions, and any pending choice.
- `diff_states(a, b, hash_names=None)` is the field-level "what changed" view — HP/block/energy deltas,
  per-zone card multiset moves, powers gained/lost/changed, enemies died, intents changed — reused by the
  TUI prediction inspector (4.4) and the predictor report card (4.3).

Hashed-lossy ids (monster/character/orb/enchant/affliction/keyword buckets — see `tokens.LOSSY_FIELDS`)
have no exact inverse, so the printer resolves them through an optional reverse map, shown as names when
present else `#bucket`:

```bash
# Scan the corpus once -> data/hash_names.json {bucket -> [names]} per vocab (collisions listed).
python -m lts2_agent.statefmt build-hash-names --corpus data/corpus
# Pretty-print / diff one corpus record (loads the map automatically if present).
python -m lts2_agent.statefmt show --corpus data/corpus --split val --index 0
python -m lts2_agent.statefmt diff --corpus data/corpus --split val --index 0
```

## Legal-action derivation (roadmap 3.3)

`lts2_agent.legal_actions.derive_option_keys` reproduces `GameHost.ListOptions` from **tokenized fields
alone**: each hand card's `canPlay` flag + target type crossed with the live hittable enemies gives the
PlayCard options (per-target for `AnyEnemy`, else untargeted); potions expand the same way by their
catalog usage/target type; `EndTurn` is always available in the player's turn; a `Choice` state's pending
offered cards give the SelectCards options. The same function will later run on decoder-predicted states
(M4); this measures it on **true** states — the upper bound.

```bash
python -m lts2_agent.legal_actions --corpus data/corpus --split val   # [--limit N]
```

Scored as set-F1 vs the recorded options by option identity (`kind` + cardId/potion + `targetCombatId`;
order-agnostic), printing overall + per-kind + per-phase exact-set/precision/recall/F1 and the top
mismatch patterns. **On 47.4k val records: exact-set 99.82%, F1 0.99936.** The residual is two enumerated
missing-information findings (the tokenizer is left unpatched — reported, not worked around): the
offered-card **order** for multi-select (`minSelect>1`) choices is lost by the sorted-multiset encoding
(so the game's exact-minimum SelectCards shortcut can't be reproduced), and a post-combat **reward screen**
whose wire `phase` is still `Combat` (`PendingRewards` is not tokenized) derives combat options instead of
the reward options.

## Protocol

The full wire spec lives in `docs/design/Lts2.Agent — Protocol.md`.
