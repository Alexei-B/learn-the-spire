# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A headless C# emulator for **Slay the Spire 2** that reuses the **real game logic** from
`sts2.dll` (no graphics/audio/engine). It boots the game's logic singletons without Godot,
then exposes read-state / list-legal-options / apply-an-option so full games can be simulated
fast and deterministically from a seed. Intended foundation for future AI-agent training (out
of scope here). Target: `net9.0`, SDK pinned by `global.json`.

The authoritative design/plan docs live in `docs/`:
- `docs/design/Sts2.Harness.md` — architecture map (read this first).
- `docs/plans/Sts2.Harness — Roadmap.md` — milestones M0–M8 with per-item status.
- `docs/plans/STS 2 Headless Emulator.md` — the original plan and research findings.

When picking up work, the "next task" is the next unchecked item in the roadmap.

## Local setup (required — the build will not work without it)

Two directories are **gitignored and must exist locally** (they hold copyrighted game content
and a large decompile — never commit them):

- `lib/` — copies of the game's binaries that `Sts2.Harness` references: `sts2.dll`,
  `0Harmony.dll`, and the non-Godot deps sts2 needs (`MonoMod.*`, `SmartFormat*`,
  `System.IO.Hashing`, `JetBrains.Annotations`, …). **`GodotSharp.dll` is intentionally NOT
  here** — the shim replaces it. `Sts2.Harness.csproj` references `lib/*.dll`.
- `refsrc/` — full ILSpy decompile of `sts2.dll` and the real `GodotSharp.dll`, for reference
  only (not compiled). **This is the primary tool for understanding game internals** — grep it
  to find types/members/call sites before writing harness code. The game also ships `sts2.xml`
  (XML docs) next to `sts2.dll`.

Game files originate from `D:\Steam\steamapps\common\Slay the Spire 2\data_sts2_windows_x86_64\`.

## Build, test, run

```sh
dotnet build learn-the-spire.sln              # build everything
dotnet test  learn-the-spire.sln              # build + run all tests (xUnit)

# Run a single test or class (substring match on FullyQualifiedName):
dotnet test --filter "FullyQualifiedName~ChoiceInjectionTests"
dotnet test --filter "FullyQualifiedName~Discovery_SurfacesChoice_AndApplyResolvesIt"
```

There is no separate lint step; `Nullable` is enabled and warnings should stay at zero.

Tests **cannot run in parallel** — parallelization is disabled assembly-wide
(`tests/.../TestCollections.cs`) because the game keeps run/combat/save state in process-wide
singletons. One run exists per process; `GameHost.StartNewRun` calls `RunManager.CleanUp()`
to tear down any prior run first.

## Architecture (3 projects)

- **`src/Sts2.GodotShim`** builds an assembly literally named **`GodotSharp`** (v4.5.1.0,
  unsigned, so sts2 binds to it by simple name) — a managed **replacement** for the real
  GodotSharp. Two kinds of content: pure value types copied verbatim from `refsrc/GodotSharp`
  (their native `NativeFuncs.godotsharp_*` calls route to a **throwing** `NativeFuncs` stub —
  a clean exception instead of an uncatchable AccessViolation), and inert hand-written facades
  for engine services (`GD`, `OS`, `FileAccess`, node hierarchy, …). **Grown empirically from
  real JIT/load errors** — never instantiate a real game Node, so the source-generator
  marshalling contract is intentionally absent. Workflow to extend it: run a test, read the
  `TypeLoad`/`MissingMethod`, add just that member, repeat.
- **`src/Sts2.Harness`** — the deliverable library. Key files:
  - `GameRuntime.cs` — one-time, process-wide headless boot. Mirrors the *logic* half of the
    game's `OneTimeInitialization` (TestMode, NonInteractiveMode, mock saves, `ModelDb.Init`,
    …), skipping all atlas/UI/ResourceLoader steps. See the ordered sequence in the file.
  - `HarmonyPatches.cs` — uses `0Harmony` to make missing localization (real tables ship only
    in the 1.9 GB `.pck` we don't have) degrade to returning the key string.
  - `GameHost.cs` — drives one run: imperative primitives (`StartNewRun`, `EnterFirstRoom`,
    `MoveTo`, `PlayCard`, `EndTurn`) plus the public API (`GetState`/`ListOptions`/`Apply`).
  - `GameState.cs` / `GameStateProjection.cs` — immutable, serializable read-model DTOs and the
    read-only projector from the live singletons.
  - `GameOption.cs` — the uniform option type (`PlayCard`/`EndTurn`/`MoveTo`/`SelectCards`);
    carries a serializable description plus internal live refs used by `Apply`.
  - `HarnessCardSelector.cs` — choice-context injection (see below).
- **`tests/Sts2.Harness.Tests`** — xUnit. Tests drive faithful end-to-end flows from a seed.

## Key mechanisms (the parts that need multiple files to understand)

- **Why it works**: the game is Godot + C#, but the *logic* (`RunState`, `CombatManager`,
  `RunManager`, `Player`, `Creature`, `CardModel`, the `Commands.*`/`GameActions.*` layers) is
  plain C#, decoupled from Godot. UI lives in `Nodes.*` (`N*.Instance` singletons) which the
  harness leaves **null** — the logic null-guards them. Content (`ModelDb`) is code-registered
  via reflection; no `.pck`/resource loading.
- **Async→sync pump**: the engine is Task-based. Actions enqueued via e.g. `CardModel.
  TryManualPlay` execute on **thread-pool continuations** (because `NonInteractiveMode.IsActive`
  is true), driven by `ActionExecutor.FinishedExecutingActions()`. The harness blocks the
  calling thread until quiescence. The enemy turn resolves on background tasks, so `EndTurn`
  waits on a `TaskCompletionSource` wired to combat events (`TurnStarted`/`CombatEnded`/
  `PlayerTurnPhaseChanged`) with a throwing safety timeout.
- **Faithful vs AutoSlay**: use the manual-play path (`CardModel.TryManualPlay`) which validates
  targeting and **pays energy** — NOT `CardCmd.AutoPlay`, which is free.
- **Choice-context injection** (`HarnessCardSelector`): mid-effect card selections
  (discover/exhaust/search/scry) go through the game's `CardSelectCmd.Selector` (`ICardSelector`)
  seam. The harness installs its own selector; when an effect requests a selection it records a
  `PendingChoice` and **blocks the effect's thread-pool task**. The combat pump
  (`GameHost.PumpCombatUntilIdleOrChoice`) waits on *whichever comes first* — queue drained or a
  choice pending — so a blocked choice returns control instead of deadlocking. The choice then
  surfaces via `GetState` (`GamePhase.Choice`) / `ListOptions` (`SelectCards` options) and is
  resolved by `Apply`, which resumes the effect on the thread pool.

## Conventions

- Match surrounding style. `ImplicitUsings` is **disabled** — add explicit `using`s. Files use
  file-scoped namespaces and target the latest C# language version.
- Grow the GodotSharp shim only as real load/JIT errors demand it; keep its behavior
  inert/managed-only and copy value types faithfully from `refsrc`.
- Keep tests seeded and deterministic; assert invariants (HP/energy/pile sanity).
- Git commit messages end with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
  Never commit anything under `lib/` or `refsrc/`.
