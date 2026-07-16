"""Transition collector: play scenario combats with cheap policies and log every decision.

Drives N parallel ``Lts2Env`` combats (a thread pool over env processes — env I/O releases the GIL,
like the trainers) and records **every** decision point — combat moves *and* mid-combat ``Choice``
selections — as a ``(state, action, next-state)`` corpus record (see :mod:`lts2_agent.corpus`).

The regimes and policies are cheap and mixable:

* **regime** ``broad`` (deckSpec ``random`` — the wide state distribution for the world model),
  ``realistic`` (deckSpec ``realistic`` — act-1-like decks), or ``mixed`` (50/50 per fight);
* **policy** ``random`` (uniform legal), ``heuristic`` (:mod:`lts2_agent.policies.heuristic` in combat,
  :mod:`lts2_agent.navigator` for choices), or ``mixed`` (per fight).

Every fight seed is ``CORPUS-<run-label>-<env>-<counter>`` — never the oracle's reserved ``PROBE-``
namespace (guarded), and collectors never issue an ``explicit`` deckSpec (the corpus writer refuses
both). A fight is written **atomically** at its end; a fight that errors (the env throws — it happens
sporadically) or exceeds the length cap is **dropped cleanly** (partial records discarded, logged to
stderr, env recreated, on to the next fight). Progress streams to the metrics dashboard as a
``kind="collect"`` run.

CLI::

    python -m lts2_agent.collect --envs 8 --fights 500 --regime mixed --policy mixed \\
        --out python/data/corpus --run-label demo

Stdout stays clean (a final one-line summary); progress/diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import corpus, navigator
from .env import Lts2Env
from .metrics import MetricsWriter
from .oracle import assert_not_probe_seed
from .policies import heuristic

# Combat option kinds the collector will pick among (masks out a stray post-combat reward option that
# can rarely appear at the exact end of a fight — same guard the scenario trainer uses).
_COMBAT_KINDS = {"PlayCard", "EndTurn", "UsePotion", "DiscardPotion"}

REGIMES = ("broad", "realistic", "mixed")
POLICIES = ("random", "heuristic", "mixed")


class _FightTruncated(Exception):
    """Raised when a fight exceeds the decision cap; its partial records are dropped."""


def make_fight_seed(run_label: str, env_idx: int, counter: int) -> str:
    """The reserved training-seed form ``CORPUS-<run-label>-<env>-<counter>`` (never ``PROBE-``)."""
    seed = f"CORPUS-{run_label}-{env_idx}-{counter}"
    return assert_not_probe_seed(seed)


def deck_spec_for_regime(regime: str) -> Dict[str, Any]:
    """The ``deckSpec`` for a concrete (non-mixed) regime."""
    if regime == "broad":
        return {"kind": "random", "cards": 15}
    if regime == "realistic":
        return {"kind": "realistic"}
    raise ValueError(f"deck_spec_for_regime needs a concrete regime, got {regime!r}")


def resolve_regime(regime: str, rng: random.Random) -> str:
    """Resolve a possibly-``mixed`` regime to ``broad``/``realistic`` for one fight."""
    if regime == "mixed":
        return rng.choice(("broad", "realistic"))
    return regime


def resolve_policy(policy: str, rng: random.Random) -> str:
    """Resolve a possibly-``mixed`` policy to ``random``/``heuristic`` for one fight."""
    if policy == "mixed":
        return rng.choice(("random", "heuristic"))
    return policy


def _legal_indices(phase: Optional[str], options: List[Dict[str, Any]]) -> List[int]:
    """Legal option indices to choose among: in combat, restrict to real combat kinds when possible."""
    if phase == "Combat":
        idxs = [i for i, o in enumerate(options) if o.get("kind") in _COMBAT_KINDS]
        if idxs:
            return idxs
    return list(range(len(options)))


def choose_action(policy: str, state: Dict[str, Any], options: List[Dict[str, Any]],
                  rng: random.Random) -> int:
    """Pick a legal option index for the given policy. ``random`` = uniform legal; ``heuristic`` =
    the reference combat ranking in combat, the scripted navigator for ``Choice`` (and any other)
    phase — the same non-combat handling the trainers use."""
    phase = state.get("phase")
    if policy == "random":
        return rng.choice(_legal_indices(phase, options))
    # heuristic
    if phase == "Combat":
        ranking = heuristic.policy(state, options)
        if ranking:
            return max(ranking, key=lambda pair: pair[1])[0]
        return _legal_indices(phase, options)[0]
    return navigator.noncombat_action(state, options)


@dataclass
class FightResult:
    """One completed fight ready to write, plus its outcome for metrics."""
    records: List[Dict[str, Any]]
    won: bool
    hp_lost: float
    act: Any
    room: Any
    character: Any
    regime: str
    policy: str


def play_fight(
    env: Lts2Env, seed: str, regime: str, policy: str, *,
    character: Optional[str], act: Optional[int], elite_pct: float, boss_pct: float,
    max_len: int, rng: random.Random,
) -> FightResult:
    """Play one isolated combat, recording every decision. Raises ``RuntimeError`` (env error) or
    :class:`_FightTruncated` (over the cap) — the caller drops the fight on either."""
    deck_spec = deck_spec_for_regime(regime)
    obs = env.reset_combat(seed=seed, character=character, elite_pct=elite_pct,
                           boss_pct=boss_pct, act=act, deck_spec=deck_spec)

    info0 = obs.get("info") or {}
    state0 = obs.get("state") or {}
    players0 = state0.get("players") or []
    character_meta = players0[0].get("character") if players0 else None
    scenario_meta = corpus.make_scenario_meta(
        deck_spec=info0.get("deckSpec"),
        removed_cards=info0.get("removedCards"),
        added_cards=info0.get("addedCards"),
        added_relics=info0.get("addedRelics"),
        added_potions=info0.get("addedPotions"),
        starter_relic_state=info0.get("starterRelicState"),
        upgraded_starter_relic=info0.get("upgradedStarterRelic"),
        act=info0.get("act"),
        room=info0.get("roomType"),
        character=character_meta,
        encounter=info0.get("encounter"),
        policy=policy,
        regime=regime,
    )

    records: List[Dict[str, Any]] = []
    t = 0
    while True:
        if obs["done"] or not obs.get("options"):
            break
        phase = (obs.get("state") or {}).get("phase")
        if phase not in ("Combat", "Choice"):
            break   # reached a non-decision live phase (fight effectively over)
        if len(records) >= max_len:
            raise _FightTruncated(f"{seed}: exceeded {max_len} decisions")

        action = choose_action(policy, obs["state"], obs["options"], rng)
        nxt = env.step(action)   # may raise RuntimeError
        records.append(corpus.make_record(
            seed=seed, scenario_meta=scenario_meta, t=t,
            state=obs["state"], options=obs["options"], action_taken=action,
            next_state=nxt.get("state"), next_options=nxt.get("options"),
            done=bool(nxt.get("done")), info=nxt.get("info"),
        ))
        t += 1
        obs = nxt

    final_info = obs.get("info") or {}
    return FightResult(
        records=records,
        won=bool(final_info.get("won")),
        hp_lost=float(final_info.get("hpLost") or 0),
        act=scenario_meta["act"],
        room=scenario_meta["room"],
        character=scenario_meta["character"],
        regime=regime,
        policy=policy,
    )


@dataclass
class CollectConfig:
    n_envs: int = 8
    fights: Optional[int] = None
    transitions: Optional[int] = None
    regime: str = "mixed"
    policy: str = "mixed"
    character: Optional[str] = None    # None = random per fight (host picks)
    act: Optional[int] = None          # None = any act
    elite_pct: float = 0.2
    boss_pct: float = 0.05
    max_fight_len: int = 90
    run_label: str = "corpus"
    flush_every: int = 20              # emit aggregate metrics every this many fights


class Collector:
    """Parallel transition collector: N env threads feeding one :class:`~corpus.CorpusWriter`."""

    def __init__(self, config: CollectConfig, writer: corpus.CorpusWriter,
                 metrics: Optional[MetricsWriter] = None,
                 host_command: Optional[Sequence[str]] = None, log=None):
        self.config = config
        self.writer = writer
        self.metrics = metrics
        self._host = list(host_command) if host_command else None
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._envs = [Lts2Env(host_command=self._host) for _ in range(config.n_envs)]
        self._counters = [itertools.count() for _ in range(config.n_envs)]
        self._lock = threading.Lock()
        self._transitions = 0
        self._fights = 0
        self._errors = 0
        self._start = time.time()
        self._last_flush_fights = 0

    def _target_reached(self) -> bool:
        cfg = self.config
        if cfg.transitions is not None and self._transitions >= cfg.transitions:
            return True
        if cfg.fights is not None and self._fights >= cfg.fights:
            return True
        return False

    def _recreate_env(self, i: int) -> None:
        try:
            self._envs[i].close()
        except Exception:
            pass
        self._envs[i] = Lts2Env(host_command=self._host)

    def _record_fight(self, res: FightResult) -> None:
        """Write one fight's records and emit its metrics (under the shared lock)."""
        n = self.writer.write_fight(res.records)
        with self._lock:
            self._transitions += n
            self._fights += 1
            fights = self._fights
            do_flush = fights - self._last_flush_fights >= self.config.flush_every
            if do_flush:
                self._last_flush_fights = fights
        if self.metrics is not None and self.metrics.enabled:
            tags = {"act": res.act, "room": res.room, "character": res.character,
                    "regime": res.regime, "policy": res.policy}
            self.metrics.emit("collect", fights, "fight.won", 1.0 if res.won else 0.0, tags=tags)
            self.metrics.emit("collect", fights, "fight.hp_lost", res.hp_lost, tags=tags)
            if do_flush:
                self._emit_aggregates(fights)

    def _emit_aggregates(self, step: int) -> None:
        with self._lock:
            transitions, fights, errors = self._transitions, self._fights, self._errors
        elapsed = max(1e-6, time.time() - self._start)
        if self.metrics is not None and self.metrics.enabled:
            self.metrics.emit("collect", step, "collect.transitions_total", float(transitions))
            self.metrics.emit("collect", step, "collect.fights_total", float(fights))
            self.metrics.emit("collect", step, "collect.errors_total", float(errors))
            self.metrics.emit("collect", step, "collect.transitions_per_s", transitions / elapsed)

    def _worker(self, i: int) -> None:
        cfg = self.config
        while not self._target_reached():
            seed = make_fight_seed(cfg.run_label, i, next(self._counters[i]))
            rng = random.Random(seed)
            regime = resolve_regime(cfg.regime, rng)
            policy = resolve_policy(cfg.policy, rng)
            try:
                res = play_fight(
                    self._envs[i], seed, regime, policy,
                    character=cfg.character, act=cfg.act,
                    elite_pct=cfg.elite_pct, boss_pct=cfg.boss_pct,
                    max_len=cfg.max_fight_len, rng=rng,
                )
            except _FightTruncated as e:
                with self._lock:
                    self._errors += 1
                self._log(f"[collect] env {i} fight dropped (truncated): {e}")
                continue
            except RuntimeError as e:
                with self._lock:
                    self._errors += 1
                self._log(f"[collect] env {i} fight dropped (env error): {e}; recreating env.")
                self._recreate_env(i)
                continue
            except Exception as e:  # noqa: BLE001 - any unexpected fight failure: drop, keep going
                with self._lock:
                    self._errors += 1
                self._log(f"[collect] env {i} fight dropped (unexpected {type(e).__name__}): {e}.")
                self._recreate_env(i)
                continue

            if not res.records:
                continue   # a degenerate immediately-over fight — nothing to log
            self._record_fight(res)
            self._log(f"[collect] env {i} {seed} regime={regime} policy={policy} "
                      f"len={len(res.records)} won={res.won} hpLost={res.hp_lost:.0f} "
                      f"[fights={self._fights} transitions={self._transitions} errors={self._errors}]")

    def run(self) -> Dict[str, Any]:
        """Collect until the fight/transition target is met; return a summary dict."""
        try:
            with ThreadPoolExecutor(max_workers=self.config.n_envs) as pool:
                futs = [pool.submit(self._worker, i) for i in range(self.config.n_envs)]
                for f in futs:
                    f.result()
        finally:
            for env in self._envs:
                try:
                    env.close()
                except Exception:
                    pass
        self._emit_aggregates(self._fights)
        elapsed = time.time() - self._start
        summary = {
            "fights": self._fights,
            "transitions": self._transitions,
            "errors": self._errors,
            "seconds": round(elapsed, 2),
            "transitionsPerSec": round(self._transitions / max(1e-6, elapsed), 2),
            "splitTotals": self.writer.totals,
        }
        self._log(f"[collect] done: {summary}")
        return summary


# --------------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lts2_agent.collect",
                                description=__doc__.split("\n")[0])
    p.add_argument("--envs", type=int, default=8, help="parallel host processes")
    p.add_argument("--fights", type=int, default=None, help="stop after this many written fights")
    p.add_argument("--transitions", type=int, default=None,
                   help="stop after this many written transitions (records)")
    p.add_argument("--regime", choices=REGIMES, default="mixed",
                   help="broad (random deck) / realistic (act-1-like) / mixed (50-50 per fight)")
    p.add_argument("--policy", choices=POLICIES, default="mixed",
                   help="random (uniform legal) / heuristic / mixed (per fight)")
    p.add_argument("--character", default="random",
                   help="character id substring, or 'random' (default) for host-random per fight")
    p.add_argument("--act", type=int, default=-1, help="restrict to act 0/1/2; -1 = any")
    p.add_argument("--elite-pct", type=float, default=0.2, help="fraction of fights that are elites")
    p.add_argument("--boss-pct", type=float, default=0.05, help="fraction of fights that are bosses")
    p.add_argument("--max-fight-len", type=int, default=90,
                   help="drop a fight exceeding this many decisions (truncated)")
    p.add_argument("--out", default=None,
                   help="corpus root dir (default: <repo>/python/data/corpus)")
    p.add_argument("--run-label", default="corpus", help="label for shard names + the metrics run")
    p.add_argument("--shard-cap", type=int, default=2000, help="records per shard before rollover")
    p.add_argument("--run-dir", default="checkpoints/runs",
                   help="metrics run directory root (dashboard reads it)")
    p.add_argument("--no-metrics", action="store_true", help="disable the metrics event stream")
    return p


def _default_out() -> str:
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "python", "data", "corpus")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fights is None and args.transitions is None:
        args.fights = 500   # a sensible default target if neither is given

    out = args.out or _default_out()
    character = None if str(args.character).lower() == "random" else args.character
    act = None if args.act is None or args.act < 0 else args.act

    config = CollectConfig(
        n_envs=args.envs, fights=args.fights, transitions=args.transitions,
        regime=args.regime, policy=args.policy, character=character, act=act,
        elite_pct=args.elite_pct, boss_pct=args.boss_pct, max_fight_len=args.max_fight_len,
        run_label=args.run_label, flush_every=20,
    )

    metrics = MetricsWriter(
        run_dir=args.run_dir, label=f"collect-{args.run_label}", argv=sys.argv,
        config=vars(args), kind="collect", enabled=not args.no_metrics,
    )
    if metrics.enabled:
        print(f"[collect] metrics -> {metrics.run_dir}", file=sys.stderr, flush=True)

    writer = corpus.CorpusWriter(out, run_label=args.run_label, shard_cap=args.shard_cap)
    print(f"[collect] corpus -> {out}", file=sys.stderr, flush=True)
    try:
        summary = Collector(config, writer, metrics).run()
    finally:
        writer.close()
        metrics.close()
    print(f"[collect] {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
