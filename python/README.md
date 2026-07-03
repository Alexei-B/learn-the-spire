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

## Protocol

The full wire spec lives in `docs/design/Lts2.Agent — Protocol.md`.
