"""Lts2 agent: a Python interface to the headless Slay the Spire 2 emulator.

Two entry points, sharing one wire protocol (:mod:`lts2_agent.protocol`):

* :class:`lts2_agent.env.Lts2Env` — a gym-style **training** environment that drives the C#
  ``Lts2.AgentHost`` process (``reset`` / ``step`` / ``close``).
* :func:`lts2_agent.decision_server.serve` — an **evaluation** decision server the TUI launches to
  get auto-play recommendations from a Python policy.

A policy trained against the environment plugs straight into the decision server unchanged.
"""

from . import protocol
from .env import Lts2Env

__all__ = ["protocol", "Lts2Env"]
