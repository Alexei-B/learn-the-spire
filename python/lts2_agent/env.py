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

import os
import shutil
import subprocess
import sys
import tempfile
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
        isolate_user_dir: bool = True,
    ) -> None:
        self.seed = seed
        self.character = character
        self.ascension = ascension
        command = list(host_command) if host_command is not None else default_host_command()

        # The host maps Godot's ``user://`` (mock saves, localization overrides) under the process's
        # temp dir. Several hosts sharing one temp dir collide on those files and a host can crash — so
        # for vectorized training each env gets its own temp dir via a private TEMP/TMP. See
        # Lts2.GodotShim/Godot/GodotPath.cs (UserDir derives from Path.GetTempPath()).
        env = None
        self._temp_dir: Optional[str] = None
        if isolate_user_dir:
            env = dict(os.environ)
            self._temp_dir = tempfile.mkdtemp(prefix="lts2env-")
            env["TEMP"] = self._temp_dir
            env["TMP"] = self._temp_dir

        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if log_stderr else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
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

    def reset_combat(
        self,
        *,
        seed: Optional[str] = None,
        character: Optional[str] = None,
        elite_pct: float = 0.2,
        boss_pct: float = 0.05,
        cards: Optional[Sequence[str]] = None,
        relics: Optional[Sequence[str]] = None,
        encounter: Optional[str] = None,
        enemy_hp: Optional[Sequence[int]] = None,
        starter_deck: bool = False,
        act: Optional[int] = None,
    ) -> dict[str, Any]:
        """Start a fresh isolated **combat scenario** and return its opening observation. By default the
        scenario is random (character/deck/relics/encounter). Pass ``cards`` (a deck of card ids) +
        ``encounter`` (an encounter type name) for a fully-specified *closed* scenario, reproducible for
        eval. The episode is a single fight: ``obs['done']`` marks the end and ``obs['info']`` carries
        ``won``/``hpLost`` (see the C# ``CombatScenario``)."""
        if seed is not None:
            self.seed = seed
        if character is not None:
            self.character = character
        message: dict[str, Any] = {
            "cmd": "reset_combat",
            "seed": self.seed,
            "character": self.character,
            "elitePct": elite_pct,
            "bossPct": boss_pct,
        }
        if starter_deck:
            message["starterDeck"] = True
        if act is not None:
            message["act"] = act
        if cards is not None:
            message["cards"] = list(cards)
            message["encounter"] = encounter
            message["relics"] = list(relics) if relics is not None else []
            if enemy_hp is not None:
                message["enemyHp"] = list(enemy_hp)
        return self._roundtrip(message)

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
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

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
