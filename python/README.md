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

### Evaluate

```sh
python -m lts2_agent.eval --policies ppo,heuristic,random --ckpt checkpoints/ppo --seeds 20
```

Reports win rate, mean/median/max floor, mean score, and mean combats survived per policy over the
same seeds — a learned policy should beat random and trend toward (then past) the heuristic.

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

## Protocol

The full wire spec lives in `docs/design/Lts2.Agent — Protocol.md`.
