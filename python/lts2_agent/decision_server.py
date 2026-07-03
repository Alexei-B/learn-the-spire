"""The evaluation-side decision server: answers ``evaluate`` requests from the TUI.

The C# ``ProcessDecisionEngine`` launches this process and, each time it wants an auto-play
recommendation, sends an ``evaluate`` request (the state + legal options) on our stdin and reads a
``scores`` response from our stdout. We call a supplied *policy* to score the options and write the
result back.

A policy is any callable ``policy(state, options) -> result`` where ``result`` is one of:

* an ``int`` — the index of the single chosen option (turned into one top-scored entry);
* a list of ``(index, score)`` tuples or ``{"index", "score", "rationale"}`` dicts — an explicit
  ranking (a subset is fine; an empty list means "decline / no recommendation").

Run it standalone with a dotted policy path::

    python -m lts2_agent.decision_server lts2_agent.policies.heuristic:policy

**stdout is reserved for protocol messages** — send any logging to stderr.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Callable, IO, Sequence

from . import protocol

Policy = Callable[[dict[str, Any], list[dict[str, Any]]], Any]


def _normalize(result: Any) -> list[dict[str, Any]]:
    """Turn a policy result into a list of ``{"index", "score", "rationale"}`` score entries."""
    if result is None:
        return []
    if isinstance(result, int):
        return [{"index": result, "score": 1.0}]
    scores: list[dict[str, Any]] = []
    for entry in result:
        if isinstance(entry, dict):
            item = {"index": int(entry["index"]), "score": float(entry.get("score", 0.0))}
            if entry.get("rationale") is not None:
                item["rationale"] = str(entry["rationale"])
            scores.append(item)
        else:
            index, score = entry  # (index, score) tuple
            scores.append({"index": int(index), "score": float(score)})
    return scores


def serve(policy: Policy, stdin: IO[str], stdout: IO[str]) -> None:
    """Read ``evaluate`` requests from ``stdin`` and write ``scores`` responses to ``stdout``."""
    while True:
        request = protocol.read_message(stdin)
        if request is None:
            return
        try:
            state = request["state"]
            options = request["options"]
            scores = _normalize(policy(state, options))
            protocol.write_message(stdout, {"scores": scores})
        except Exception as exc:  # never crash the server on one bad request
            print(f"[decision-server] error: {exc}", file=sys.stderr, flush=True)
            protocol.write_message(stdout, {"scores": []})


def load_policy(spec: str) -> Policy:
    """Load a policy from a ``module.path:attr`` spec (defaults to ``:policy``)."""
    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr or "policy")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lts2 evaluation decision server.")
    parser.add_argument(
        "policy",
        nargs="?",
        default="lts2_agent.policies.heuristic:policy",
        help="Policy spec as module.path:attr (default: lts2_agent.policies.heuristic:policy).",
    )
    args = parser.parse_args(argv)
    policy = load_policy(args.policy)
    serve(policy, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
