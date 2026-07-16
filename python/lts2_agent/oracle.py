"""Oracle prober: replay-based ground-truth next states for a frozen probe set.

The world-model's *predictor* will predict the next observation given a state and an action. To
score it we need ground truth: for a fixed set of combat *positions*, the true next observation for
**every** legal action. The emulator is deterministic — same seed + same reset params + same action
sequence reproduces a fight exactly — but there are no mid-combat snapshots, so a position is reached
by **replaying an action prefix** from a seeded combat start. That is the whole idea here:

* a **probe** freezes a reproducible position: reset params + an action prefix (option indices) +
  captured metadata (act / room / character / turn / option-count) for stratification;
* the **oracle** replays each probe, enumerates its legal options, and for each option index replays
  ``reset + prefix + [i]`` to record the resulting full observation — the ground-truth next state.

**Evaluation-only, by construction.** Every probe seed lives in a reserved namespace: it is prefixed
with ``PROBE-``. Training collectors MUST NEVER use that prefix, so probe fights can never leak into a
training corpus. :func:`validate_probe_seed` enforces the rule; the builder always prefixes.

CLI (all against the built ``Lts2.AgentHost`` — build it first, see :mod:`lts2_agent.env`)::

    python -m lts2_agent.oracle build  --n 40  --out lts2_agent/data/probes.json
    python -m lts2_agent.oracle run    --probes lts2_agent/data/probes.json --out shard.jsonl.gz --envs 4
    python -m lts2_agent.oracle verify --probes lts2_agent/data/probes.json --sample 20

Progress and diagnostics go to **stderr**; stdout stays clean. Records/probes are written to files.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import queue
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .env import Lts2Env

# --------------------------------------------------------------------------------------------------
# Seed namespace — the anti-overfit guard.
# --------------------------------------------------------------------------------------------------

PROBE_SEED_PREFIX = "PROBE-"
"""Reserved seed prefix for evaluation-only probe fights. Training collectors must never use it."""


def is_probe_seed(seed: str) -> bool:
    """True iff ``seed`` is in the reserved probe namespace."""
    return isinstance(seed, str) and seed.startswith(PROBE_SEED_PREFIX)


def validate_probe_seed(seed: str) -> str:
    """Return ``seed`` unchanged if it is a probe seed; otherwise raise loudly.

    The oracle calls this on every probe it builds or replays, so a non-``PROBE-`` seed can never be
    frozen into a probe set — probes stay structurally excluded from training seeds.
    """
    if not is_probe_seed(seed):
        raise ValueError(
            f"Probe seed {seed!r} is not in the reserved {PROBE_SEED_PREFIX!r} namespace. "
            "Probe seeds must be prefixed so they can never collide with a training seed."
        )
    return seed


def assert_not_probe_seed(seed: str) -> str:
    """Guard for training collectors: reject a probe seed leaking into training data."""
    if is_probe_seed(seed):
        raise ValueError(
            f"Seed {seed!r} uses the reserved {PROBE_SEED_PREFIX!r} probe namespace and is "
            "evaluation-only. Training data must never use a probe seed."
        )
    return seed


# --------------------------------------------------------------------------------------------------
# Probe: a reproducible frozen position.
# --------------------------------------------------------------------------------------------------

# reset_params keys (camelCase, matching the protocol/doc). Mapped to env.reset_combat kwargs below.
_RESET_PARAM_KEYS = ("seed", "character", "elitePct", "bossPct", "starterDeck", "act")


@dataclass
class Probe:
    """A reproducible combat position: reset params + an action prefix, plus captured metadata.

    Replaying ``env.reset_combat(**reset_kwargs) `` then stepping ``action_prefix`` in order reaches the
    exact position. ``meta`` is captured at build time for stratification only (act/room/character/…).
    """

    probe_id: str
    reset_params: Dict[str, Any]
    action_prefix: List[int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_probe_seed(self.reset_params["seed"])

    def reset_kwargs(self) -> Dict[str, Any]:
        """Translate the stored (camelCase) reset params into :meth:`Lts2Env.reset_combat` kwargs."""
        rp = self.reset_params
        return dict(
            seed=rp["seed"],
            character=rp.get("character"),
            elite_pct=rp["elitePct"],
            boss_pct=rp["bossPct"],
            starter_deck=bool(rp.get("starterDeck", False)),
            act=rp.get("act"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probeId": self.probe_id,
            "resetParams": {k: self.reset_params.get(k) for k in _RESET_PARAM_KEYS},
            "actionPrefix": list(self.action_prefix),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Probe":
        return cls(
            probe_id=d["probeId"],
            reset_params=dict(d["resetParams"]),
            action_prefix=list(d["actionPrefix"]),
            meta=dict(d.get("meta", {})),
        )


def save_probe_set(path: str, probes: Sequence[Probe], generator: Dict[str, Any]) -> None:
    """Write a sorted probe set with a header block ``{createdAt, count, generator}`` to ``path``."""
    ordered = sorted(probes, key=lambda p: p.probe_id)
    doc = {
        "header": {
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "count": len(ordered),
            "generator": generator,
        },
        "probes": [p.to_dict() for p in ordered],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_probe_set(path: str) -> Tuple[Dict[str, Any], List[Probe]]:
    """Read a probe set file; return ``(header, probes)``."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    header = doc.get("header", {})
    probes = [Probe.from_dict(p) for p in doc.get("probes", [])]
    return header, probes


# --------------------------------------------------------------------------------------------------
# Observation cleanup + metadata capture.
# --------------------------------------------------------------------------------------------------

_OBS_KEYS = ("state", "options", "done", "info")


def clean_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the wire observation fields (drop transport cruft like ``_bytes``)."""
    return {k: obs[k] for k in _OBS_KEYS if k in obs}


def capture_meta(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stratification metadata (act / room / character / turn / phase / optionCount)."""
    state = obs.get("state") or {}
    info = obs.get("info") or {}
    players = state.get("players") or []
    p0 = players[0] if players else {}
    combat = p0.get("combatState") or {}
    return {
        "act": info.get("act", state.get("actIndex")),
        "roomType": info.get("roomType"),
        "character": p0.get("character"),
        "turn": combat.get("turnNumber"),
        "phase": state.get("phase"),
        "optionCount": len(obs.get("options") or []),
    }


def _is_freezable(obs: Dict[str, Any]) -> bool:
    """A position is freezable iff the fight is live and offers a real decision (>=2 options,
    phase Combat or Choice)."""
    if obs.get("done") or not obs.get("options"):
        return False
    if len(obs["options"]) < 2:
        return False
    return (obs.get("state") or {}).get("phase") in ("Combat", "Choice")


# --------------------------------------------------------------------------------------------------
# Probe-set builder.
# --------------------------------------------------------------------------------------------------

_ROOM_CLASSES = ("monster", "elite", "boss")


def plan_probe(rng: random.Random, index: int, characters: Optional[Sequence[str]]) -> Tuple[Dict[str, Any], int]:
    """Deterministically choose reset params + a prefix length for probe ``index`` (pure — no env).

    Spans acts 0-2, a monster/elite/boss room mix (via the elite/boss pct knobs), all characters
    (a random character or, half the time, ``None`` to let the host randomize), broad-vs-starter
    decks, and a varied replay depth (0-15 steps). The probe seed is always ``PROBE-``-prefixed.
    """
    room = rng.choice(_ROOM_CLASSES)
    act = rng.randint(0, 2)
    starter = rng.random() < 0.3
    if characters and rng.random() < 0.5:
        character: Optional[str] = rng.choice(list(characters))
    else:
        character = None
    steps = rng.randint(0, 15)
    reset_params = {
        "seed": f"{PROBE_SEED_PREFIX}{index:05d}",
        "character": character,
        "elitePct": 1.0 if room == "elite" else 0.0,
        "bossPct": 1.0 if room == "boss" else 0.0,
        "starterDeck": starter,
        "act": act,
    }
    validate_probe_seed(reset_params["seed"])
    return reset_params, steps


def _try_build_one(
    env: Lts2Env, index: int, reset_params: Dict[str, Any], steps: int, rng: random.Random
) -> Optional[Tuple[Probe, str]]:
    """Replay reset + up-to-``steps`` uniformly-random legal actions; freeze the position reached.

    Returns ``(probe, state_json)`` — the frozen probe plus its canonical serialized state (the
    reproducibility reference) — or ``None`` (skip) if the fight ends first or the final position is
    not freezable.
    """
    kwargs = dict(
        seed=reset_params["seed"], character=reset_params.get("character"),
        elite_pct=reset_params["elitePct"], boss_pct=reset_params["bossPct"],
        starter_deck=bool(reset_params.get("starterDeck", False)), act=reset_params.get("act"),
    )
    try:
        obs = env.reset_combat(**kwargs)
    except RuntimeError:
        return None
    prefix: List[int] = []
    for _ in range(steps):
        if obs.get("done") or not obs.get("options"):
            break
        if (obs.get("state") or {}).get("phase") not in ("Combat", "Choice"):
            break
        idx = rng.randrange(len(obs["options"]))
        prefix.append(idx)
        try:
            obs = env.step(idx)
        except RuntimeError:
            return None
    if not _is_freezable(obs):
        return None
    probe = Probe(
        probe_id=f"probe-{index:05d}",
        reset_params=reset_params,
        action_prefix=prefix,
        meta=capture_meta(obs),
    )
    return probe, _state_json(clean_obs(obs))


def _is_reproducible(env: Lts2Env, probe: Probe, ref_state_json: str, n_check: int) -> bool:
    """Replay ``probe`` ``n_check`` times on a **separate** env (fresh host process) and require every
    replayed state to be byte-identical to ``ref_state_json``.

    Some deep-replay fights are genuinely non-reproducible (an RNG not reseeded per fight, or
    async-combat-pump ordering) — the position is then ill-defined and useless as ground truth. This
    gate rejects such candidates so every committed probe is a *stable*, cross-process-reproducible
    position. Returns ``False`` on any mismatch or host error."""
    for _ in range(n_check):
        try:
            got = _state_json(_replay_position(env, probe))
        except RuntimeError:
            return False
        if got != ref_state_json:
            return False
    return True


def build_probes(
    n: int,
    master_seed: str,
    host_command: Optional[Sequence[str]] = None,
    characters: Optional[Sequence[str]] = None,
    max_attempts_factor: int = 12,
    reproduce_checks: int = 8,
    log: Optional[Callable[[str], None]] = None,
) -> List[Probe]:
    """Build ``n`` probes with a seeded RNG (reproducible). Deterministic regardless of order because
    each attempt is independently seeded from ``master_seed``.

    Every candidate must pass a **reproducibility gate**: it is replayed ``reproduce_checks`` times on a
    *second, independent* host process and kept only if every replay is byte-identical to the frozen
    state. This rejects the deep-replay fights that are genuinely non-reproducible — both the fully
    chaotic ones and the *flaky* ones that only diverge occasionally (so more checks catch more) — so
    every committed probe is a stable, cross-process ground-truth position. Set ``reproduce_checks=0``
    to disable (not recommended)."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    probes: List[Probe] = []
    attempt = 0
    dropped = 0
    max_attempts = max(n * max_attempts_factor, n + 1)
    host = list(host_command) if host_command else None
    env = Lts2Env(host_command=host)                         # builds positions
    checker = Lts2Env(host_command=host) if reproduce_checks else None  # independent reproducibility check
    try:
        while len(probes) < n and attempt < max_attempts:
            rng = random.Random(f"{master_seed}:{attempt}")
            reset_params, steps = plan_probe(rng, attempt, characters)
            try:
                built = _try_build_one(env, attempt, reset_params, steps, rng)
            except RuntimeError as e:
                # The host process died; recreate it and treat this attempt as a skip.
                log(f"[oracle:build] attempt {attempt} host error ({e}); recreating env.")
                env.close()
                env = Lts2Env(host_command=host)
                built = None
            if built is not None:
                probe, ref_json = built
                if checker is not None and not _is_reproducible(checker, probe, ref_json, reproduce_checks):
                    dropped += 1
                    log(f"[oracle:build] attempt {attempt} {probe.probe_id} NOT reproducible "
                        f"(prefix={len(probe.action_prefix)}); dropped.")
                    # A crashed checker leaves it unusable; recreate defensively.
                    if checker._proc.poll() is not None:  # type: ignore[attr-defined]
                        checker.close()
                        checker = Lts2Env(host_command=host)
                else:
                    probes.append(probe)
                    m = probe.meta
                    log(f"[oracle:build] {len(probes)}/{n} probe={probe.probe_id} "
                        f"act={m.get('act')} room={m.get('roomType')} char={m.get('character')} "
                        f"turn={m.get('turn')} opts={m.get('optionCount')} prefix={len(probe.action_prefix)}")
            attempt += 1
    finally:
        env.close()
        if checker is not None:
            checker.close()
    log(f"[oracle:build] built {len(probes)}/{n} in {attempt} attempts "
        f"({dropped} dropped as non-reproducible).")
    if len(probes) < n:
        log(f"[oracle:build] WARNING: only built {len(probes)}/{n} probes.")
    probes.sort(key=lambda p: p.probe_id)
    return probes


# --------------------------------------------------------------------------------------------------
# Oracle runner: ground-truth next state for every legal action at each probe.
# --------------------------------------------------------------------------------------------------


def make_shard_record(
    probe_id: str,
    position: Optional[Dict[str, Any]],
    results: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one shard record. Either a full record (position + per-action results) or, when the
    position replay itself failed, ``{probeId, error}``."""
    if error is not None:
        return {"probeId": probe_id, "error": error}
    rec: Dict[str, Any] = {"probeId": probe_id, "position": position, "results": results or []}
    if meta is not None:
        rec["meta"] = meta
    return rec


def validate_shard_record(rec: Dict[str, Any]) -> None:
    """Assert a shard record matches the schema; raise ``ValueError`` on the first violation."""
    if "probeId" not in rec or not isinstance(rec["probeId"], str):
        raise ValueError("shard record needs a string 'probeId'")
    if "error" in rec:
        if not isinstance(rec["error"], str):
            raise ValueError("'error' must be a string")
        return
    if "position" not in rec or "results" not in rec:
        raise ValueError("a non-error shard record needs 'position' and 'results'")
    if not isinstance(rec["results"], list):
        raise ValueError("'results' must be a list")
    for r in rec["results"]:
        if "action" not in r or not isinstance(r["action"], int):
            raise ValueError("each result needs an int 'action'")
        if "obs" not in r and "error" not in r:
            raise ValueError("each result needs either 'obs' or 'error'")


def _replay_position(env: Lts2Env, probe: Probe) -> Dict[str, Any]:
    """Replay reset + the full prefix; return the cleaned position observation. Raises on env error."""
    obs = env.reset_combat(**probe.reset_kwargs())
    for idx in probe.action_prefix:
        obs = env.step(idx)
    return clean_obs(obs)


def _replay_and_step(env: Lts2Env, probe: Probe, action: int) -> Dict[str, Any]:
    """Replay reset + prefix + one more action; return the cleaned resulting observation."""
    obs = env.reset_combat(**probe.reset_kwargs())
    for idx in probe.action_prefix:
        obs = env.step(idx)
    obs = env.step(action)
    return clean_obs(obs)


class _EnvHolder:
    """A recreatable env handle for a worker thread (mirrors the trainers' env-recreate-on-crash)."""

    def __init__(self, host_command: Optional[Sequence[str]]):
        self._host_command = list(host_command) if host_command else None
        self.env = Lts2Env(host_command=self._host_command)

    def recreate(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        self.env = Lts2Env(host_command=self._host_command)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def _oracle_one(holder: _EnvHolder, probe: Probe) -> Dict[str, Any]:
    """Produce the shard record for one probe: position + ground-truth next obs per legal action."""
    try:
        position = _replay_position(holder.env, probe)
    except RuntimeError as e:
        holder.recreate()
        return make_shard_record(probe.probe_id, None, error=f"position replay failed: {e}")

    n_opts = len(position.get("options") or [])
    results: List[Dict[str, Any]] = []
    for i in range(n_opts):
        try:
            nxt = _replay_and_step(holder.env, probe, i)
            results.append({"action": i, "obs": nxt})
        except RuntimeError as e:
            # Hard env failure: recreate and record the error, then continue with the rest.
            holder.recreate()
            results.append({"action": i, "error": f"env error: {e}"})
        except Exception as e:  # noqa: BLE001 - tolerate any per-action failure
            results.append({"action": i, "error": str(e)})
    return make_shard_record(probe.probe_id, position, results, meta=probe.meta)


def run_oracle(
    probes: Sequence[Probe],
    out_path: str,
    host_command: Optional[Sequence[str]] = None,
    n_envs: int = 1,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Replay every probe and write a gzip-JSONL shard (one record per probe). Parallel across
    ``n_envs`` host processes (a thread pool over env instances — env I/O releases the GIL).

    Returns a small summary ``{count, errors, actions, seconds}``."""
    import time

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    probes = list(probes)
    n_envs = max(1, min(n_envs, len(probes) or 1))
    work: "queue.Queue[Optional[Probe]]" = queue.Queue()
    for p in probes:
        work.put(p)

    write_lock = threading.Lock()
    counters = {"done": 0, "errors": 0, "actions": 0}
    total = len(probes)
    start = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gz = gzip.open(out_path, "wt", encoding="utf-8")

    def worker() -> None:
        holder = _EnvHolder(host_command)
        try:
            while True:
                try:
                    probe = work.get_nowait()
                except queue.Empty:
                    return
                t0 = time.time()
                rec = _oracle_one(holder, probe)
                line = json.dumps(rec, separators=(",", ":"))
                with write_lock:
                    gz.write(line)
                    gz.write("\n")
                    counters["done"] += 1
                    if "error" in rec:
                        counters["errors"] += 1
                    else:
                        counters["actions"] += len(rec.get("results", []))
                    done = counters["done"]
                dt = time.time() - t0
                log(f"[oracle:run] {done}/{total} {probe.probe_id} "
                    f"opts={len(rec.get('results', []))} {dt:.2f}s"
                    + (f" ERROR={rec['error']}" if "error" in rec else ""))
        finally:
            holder.close()

    with ThreadPoolExecutor(max_workers=n_envs) as pool:
        futs = [pool.submit(worker) for _ in range(n_envs)]
        for f in futs:
            f.result()
    gz.close()

    secs = time.time() - start
    summary = {
        "count": counters["done"],
        "errors": counters["errors"],
        "actions": counters["actions"],
        "seconds": round(secs, 2),
        "secPerProbe": round(secs / max(1, counters["done"]), 3),
    }
    log(f"[oracle:run] done: {summary}")
    return summary


# --------------------------------------------------------------------------------------------------
# Determinism verification.
# --------------------------------------------------------------------------------------------------


def _state_json(obs: Dict[str, Any]) -> str:
    return json.dumps(obs.get("state"), sort_keys=True, separators=(",", ":"))


def verify_positions(
    probes: Sequence[Probe],
    host_command: Optional[Sequence[str]] = None,
    n_envs: int = 2,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Replay each probe's position **twice** and assert the serialized state JSON is byte-identical.

    Doubles as the CP2 determinism spot-check. Returns ``{checked, ok, mismatches, errors,
    mismatchIds}``; a non-empty ``mismatchIds`` means determinism is broken and is reported loudly."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    probes = list(probes)
    n_envs = max(1, min(n_envs, len(probes) or 1))
    work: "queue.Queue[Optional[Probe]]" = queue.Queue()
    for p in probes:
        work.put(p)

    lock = threading.Lock()
    stats = {"checked": 0, "ok": 0, "mismatches": 0, "errors": 0}
    mismatch_ids: List[str] = []
    total = len(probes)

    def worker() -> None:
        holder = _EnvHolder(host_command)
        try:
            while True:
                try:
                    probe = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    a = _replay_position(holder.env, probe)
                    b = _replay_position(holder.env, probe)
                except RuntimeError as e:
                    holder.recreate()
                    with lock:
                        stats["checked"] += 1
                        stats["errors"] += 1
                    log(f"[oracle:verify] {probe.probe_id} REPLAY ERROR: {e}")
                    continue
                same = _state_json(a) == _state_json(b)
                with lock:
                    stats["checked"] += 1
                    if same:
                        stats["ok"] += 1
                    else:
                        stats["mismatches"] += 1
                        mismatch_ids.append(probe.probe_id)
                    checked = stats["checked"]
                log(f"[oracle:verify] {checked}/{total} {probe.probe_id} "
                    + ("ok" if same else "*** MISMATCH ***"))
        finally:
            holder.close()

    with ThreadPoolExecutor(max_workers=n_envs) as pool:
        futs = [pool.submit(worker) for _ in range(n_envs)]
        for f in futs:
            f.result()

    result = dict(stats, mismatchIds=sorted(mismatch_ids))
    if mismatch_ids:
        log(f"[oracle:verify] !!! DETERMINISM BROKEN: {len(mismatch_ids)} mismatch(es): "
            f"{sorted(mismatch_ids)}")
    else:
        log(f"[oracle:verify] determinism OK: {result}")
    return result


# --------------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    characters = [c.strip() for c in args.characters.split(",") if c.strip()] if args.characters else None
    probes = build_probes(
        n=args.n,
        master_seed=args.master_seed,
        characters=characters,
        max_attempts_factor=args.max_attempts_factor,
        reproduce_checks=args.reproduce_checks,
    )
    generator = {
        "n": args.n,
        "built": len(probes),
        "masterSeed": args.master_seed,
        "characters": characters,
        "maxAttemptsFactor": args.max_attempts_factor,
        "reproduceChecks": args.reproduce_checks,
        "seedPrefix": PROBE_SEED_PREFIX,
    }
    save_probe_set(args.out, probes, generator)
    print(f"[oracle:build] wrote {len(probes)} probes to {args.out}", file=sys.stderr)
    return 0 if len(probes) == args.n else 1


def _cmd_run(args: argparse.Namespace) -> int:
    _header, probes = load_probe_set(args.probes)
    if args.limit is not None:
        probes = probes[: args.limit]
    if args.verify:
        sample = probes[: args.sample] if args.sample else probes
        vres = verify_positions(sample, n_envs=args.envs)
        if vres["mismatchIds"]:
            print(f"[oracle:run] verification found mismatches: {vres['mismatchIds']}", file=sys.stderr)
    summary = run_oracle(probes, args.out, n_envs=args.envs)
    print(f"[oracle:run] shard written to {args.out}: {summary}", file=sys.stderr)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    _header, probes = load_probe_set(args.probes)
    if args.sample:
        probes = probes[: args.sample]
    res = verify_positions(probes, n_envs=args.envs)
    return 1 if res["mismatchIds"] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lts2_agent.oracle", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="freeze a probe set (reproducible)")
    pb.add_argument("--n", type=int, default=40, help="number of probes to build")
    pb.add_argument("--out", required=True, help="output probes.json path")
    pb.add_argument("--master-seed", default="PROBE-BUILD-0", help="RNG master seed (reproducible)")
    pb.add_argument("--characters", default=None,
                    help="comma-separated character ids to sample from (default: omit -> host random)")
    pb.add_argument("--max-attempts-factor", type=int, default=12,
                    help="cap attempts at n*this (fights that end early / non-reproducible are skipped)")
    pb.add_argument("--reproduce-checks", type=int, default=8,
                    help="replay each candidate this many times on a second host and require byte-identical "
                         "state; 0 disables the reproducibility gate (not recommended)")
    pb.set_defaults(func=_cmd_build)

    pr = sub.add_parser("run", help="replay probes -> ground-truth next states (gzip JSONL shard)")
    pr.add_argument("--probes", required=True, help="probes.json path")
    pr.add_argument("--out", required=True, help="output shard path (gzip JSONL)")
    pr.add_argument("--envs", type=int, default=1, help="parallel host processes")
    pr.add_argument("--limit", type=int, default=None, help="only run the first N probes")
    pr.add_argument("--verify", action="store_true", help="also run the double-replay determinism check first")
    pr.add_argument("--sample", type=int, default=0, help="with --verify: how many probes to check (0 = all)")
    pr.set_defaults(func=_cmd_run)

    pv = sub.add_parser("verify", help="double-replay determinism spot-check")
    pv.add_argument("--probes", required=True, help="probes.json path")
    pv.add_argument("--envs", type=int, default=2, help="parallel host processes")
    pv.add_argument("--sample", type=int, default=0, help="how many probes to check (0 = all)")
    pv.set_defaults(func=_cmd_verify)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
