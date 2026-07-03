"""The Lts2 agent wire protocol: one JSON object per line over a byte stream.

This mirrors ``src/Lts2.Agent/Wire`` on the C# side. Both directions share the same encoding:

* an **observation** is the game ``state`` serialized as-is plus the legal ``options`` (an action is
  the *index* of an option in that list; a "choose N of M" card choice uses ``cardIndices`` instead);
* the training driver sends ``reset`` / ``step`` / ``close`` commands and receives observations;
* the evaluation decision server receives ``evaluate`` requests and returns ``scores``.

Keeping the encoding identical means a policy trained against the environment server
(:mod:`lts2_agent.env`) plugs straight into the TUI behind the decision server
(:mod:`lts2_agent.decision_server`).
"""

from __future__ import annotations

import json
from typing import Any, IO, Optional

PROTOCOL_VERSION = 1


def write_message(stream: IO[str], message: dict[str, Any]) -> None:
    """Write one message as a single JSON line and flush it immediately."""
    stream.write(json.dumps(message, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def read_message(stream: IO[str]) -> Optional[dict[str, Any]]:
    """Read the next JSON-line message, or ``None`` at end of stream."""
    line = stream.readline()
    if line == "":
        return None
    line = line.strip()
    if not line:
        # Blank line: skip and try the next one.
        return read_message(stream)
    return json.loads(line)


def legal_action_count(observation: dict[str, Any]) -> int:
    """Number of legal options in an observation (the action space size for this step)."""
    return len(observation.get("options", []))


def is_terminal(observation: dict[str, Any]) -> bool:
    """Whether the run has ended (no further actions are possible)."""
    return bool(observation.get("done", False))
