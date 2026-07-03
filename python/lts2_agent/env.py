"""A gym-style environment that drives the C# headless emulator over the line protocol.

:class:`Lts2Env` spawns the ``Lts2.AgentHost`` process and talks to it on its stdio: ``reset`` starts
a fresh run, ``step`` applies an action, and each returns the full observation (state + legal options
+ terminal flag + a compact ``info`` block). **Reward is not baked in** — every ``step`` returns the
raw observation and its ``info`` (score, per-player hp, floor, victory), and the training loop derives
whatever reward signal it wants (see ``examples/train_stub.py``).

One process hosts one run at a time (the game keeps run state in process-wide singletons), so a
vectorized trainer should spawn several :class:`Lts2Env` instances.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from . import protocol

Action = Union[int, Sequence[int]]


def default_host_command() -> list[str]:
    """Locate the built ``Lts2.AgentHost`` and return a command to run it with ``dotnet``.

    Searches ``src/Lts2.AgentHost/bin`` under the repo root (this file is at
    ``python/lts2_agent/env.py``). Build it first with
    ``dotnet build src/Lts2.AgentHost/Lts2.AgentHost.csproj``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = repo_root / "src" / "Lts2.AgentHost" / "bin"
    candidates = sorted(bin_dir.glob("**/Lts2.AgentHost.dll"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find Lts2.AgentHost.dll under {bin_dir}. "
            "Build it with: dotnet build src/Lts2.AgentHost/Lts2.AgentHost.csproj"
        )
    # Prefer the most recently built.
    dll = max(candidates, key=lambda p: p.stat().st_mtime)
    return ["dotnet", str(dll)]


class Lts2Env:
    """A single Slay the Spire 2 run as a step-able environment."""

    def __init__(
        self,
        host_command: Optional[Sequence[str]] = None,
        *,
        seed: str = "AGENT",
        character: Optional[str] = None,
        ascension: int = 0,
        log_stderr: bool = False,
    ) -> None:
        self.seed = seed
        self.character = character
        self.ascension = ascension
        command = list(host_command) if host_command is not None else default_host_command()
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if log_stderr else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def reset(
        self,
        *,
        seed: Optional[str] = None,
        character: Optional[str] = None,
        ascension: Optional[int] = None,
    ) -> dict[str, Any]:
        """Start a fresh run and return the opening observation."""
        if seed is not None:
            self.seed = seed
        if character is not None:
            self.character = character
        if ascension is not None:
            self.ascension = ascension
        return self._roundtrip(
            {
                "cmd": "reset",
                "seed": self.seed,
                "character": self.character,
                "ascension": self.ascension,
            }
        )

    def step(self, action: Action) -> dict[str, Any]:
        """Apply an action and return the resulting observation.

        ``action`` is either an ``int`` (the index of a legal option) or a sequence of ``int``
        (the card indices for a "choose N of M" choice — valid only while a card choice is pending).
        """
        if isinstance(action, int):
            message = {"cmd": "step", "index": action}
        else:
            message = {"cmd": "step", "cardIndices": list(action)}
        return self._roundtrip(message)

    def close(self) -> None:
        """Close the run and terminate the host process."""
        if self._proc.poll() is None:
            try:
                protocol.write_message(self._proc.stdin, {"cmd": "close"})
                protocol.read_message(self._proc.stdout)
            except (BrokenPipeError, ValueError):
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _roundtrip(self, message: dict[str, Any]) -> dict[str, Any]:
        protocol.write_message(self._proc.stdin, message)
        response = protocol.read_message(self._proc.stdout)
        if response is None:
            raise RuntimeError("Environment host closed its output stream unexpectedly.")
        if "error" in response:
            raise RuntimeError(f"Environment error: {response['error']}")
        return response

    def __enter__(self) -> "Lts2Env":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
