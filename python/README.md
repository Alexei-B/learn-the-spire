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
  act/floor/ascension/score/energy/stars/turn/gold/…); one **card** token per card in hand, draw, discard,
  exhaust, and the offered cards of a pending choice (zone id + card-catalog index + the live `CardView`
  dynamic fields: cost/costsX/starCost/damage/baseDamage/block/baseBlock/summon/upgraded/canPlay/
  replayCount/enchant/affliction + a hashed multi-hot of `addedKeywords`); one **creature** token per
  player/Osty/enemy (hp/maxHp/block/active + identity); **power** tokens (power-catalog index + amount,
  parented to a creature); **intent** tokens (type/damage/hits, parented to an enemy); **orb**, **relic**
  (relic-catalog index), **potion** (potion-catalog index, per belt slot incl. empty), and a **pending-
  choice** token (min/max select + upgrade flag). Fixed-shape padded arrays + boolean masks batch cleanly.
- **The draw pile (and every card zone) is an unordered MULTISET.** Card tokens within a zone are sorted
  by their full content tuple, so the wire's shuffle order can *never* leak. Two shuffles of the same pile
  produce byte-identical tokens (`test_tokens.test_draw_pile_is_unordered_multiset`).
- **Numerics use symlog** (`sign(x)·log1p|x|`, DreamerV3-style) with a ±`NUM_CLIP` (1e5) clamp — bounded
  (no encoder blow-ups) and **exactly invertible** for integer game quantities, which is what makes the
  round-trip validator exact. The clamp only saturates the game's `999999999` "no maximum" select sentinel
  and any pathological scaling outlier. Every categorical is a **catalog index** (cards/powers/relics/
  potions — exact inverse) or a small **fixed enum** (zones, token/intent/target/card types, phases —
  enumerated from `GameState.cs`).
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

## Protocol

The full wire spec lives in `docs/design/Lts2.Agent — Protocol.md`.
