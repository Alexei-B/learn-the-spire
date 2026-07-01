# Sts2.Tui — full-screen terminal client for the headless harness

A **full-screen, terminal-owning** client (think ncurses / Dwarf Fortress) that plays single-player
**Slay the Spire 2** runs, driving the `Sts2.Harness` library (the real game logic, no Godot). It
exists for **manual testing / poking at the harness**: character select → ascension → seed → a
complete run through the Neow ancient event, the map, combats, events, rewards, treasures, rest
sites and shops.

Built with [Terminal.Gui](https://github.com/gui-cs/Terminal.Gui) **v2** — it takes over the whole
terminal (alternate screen buffer, redraw-in-place, windows + list views + menus), rather than
printing scrolling text. The board is drawn with a soft true-colour theme (`Theme.cs`).

Two non-obvious things make Terminal.Gui v2 coexist with the headless game:
- **Init order** (`Program.RunTui`): Terminal.Gui's `ConfigurationManager` scans every *already-
  loaded* assembly's types at `Application.Init()`. Our GodotSharp shim is intentionally incomplete
  (its Godot source-generator types aren't all defined), so enumerating it throws
  `ReflectionTypeLoadException`. We therefore call `Application.Init()` **before** booting the game
  (which loads GodotSharp/sts2) — the one-time scan runs while those assemblies aren't loaded.
- **Logging** stays off the screen via the shim's redirectable `GD.Out`/`GD.Err` (see below).

## Run it

```sh
dotnet run --project src/Sts2.Tui
```

## Localized names & descriptions (optional but recommended)

The board, the action list, and a **Details** panel show real card/relic/potion/event names and
descriptions (with the actual numbers — "Deal 8 damage. Apply 2 Vulnerable.") via the
**`Sts2.Localization`** library. That text lives only in the game `.pck`, so extract it once per
clone (it lands under the gitignored `lib/`):

```sh
pwsh scripts/extract-localization.ps1   # uses GDRE Tools to pull res://localization/eng/*.json
```

If you skip this, the TUI still works — it just shows model ids (e.g. `BASH`) instead of names.
The harness itself never depends on this; only the TUI (via the library) does.

A **New Run** dialog opens first (character, ascension 0–10, seed). Then the main screen:

```
 Game   View
┌ Board — Combat ──────────────────────────────────┐┌ Piles ─────────────┐
│ Act 1 · Floor 2 · 80/80 HP · 99g …               ││ DRAW (5)           │
│ ┌ IRONCLAD ─────────┐  ┌ #1 Leaf Slime (S) ────┐ ││   Strike x4  Bash  │
│ │ ███████████ 80/80 │  │ ██████████ 12/12      │ ││ DISCARD (0)        │
│ │ Energy 3/3 Hand 5 │  │ Intent: Attack 3      │ ││ EXHAUST (0)        │
│ └───────────────────┘  └───────────────────────┘ ││                    │
│ …                                                 ││                    │
└───────────────────────────────────────────────────┘└────────────────────┘
┌ Decisions  (↑↓ · 1-9 · Enter) ─────────────────────────────────────────────┐
│ [1] (1) Strike → #1                                                         │
│       Deal 6 damage.                                                        │
│ [2] (1) Defend                                                              │
│       Gain 5 Block.                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **Board** (top-left) — the live state for the phase. In **combat**: allies (players + Osty) on the
  left and enemies on the right, each a bordered box with a coloured **health bar** (red current HP,
  dark lost HP, green = HP poison will remove this turn, purple = doom threshold; the border turns
  green/purple if they'd die to poison/doom this turn, grey if they have block), an info line
  (energy + hand / enemy intent), and its powers. On the **map** screen: the act map with connections
  drawn between rooms.
- **Side panel** (top-right) — the act **map** (with connections) on every screen, except in combat
  where it shows your **draw / discard / exhaust piles**.
- **Decisions** (bottom) — the legal options with their localized descriptions inline. Move with
  **↑/↓**, **1–9** to quick-pick, **Enter** to apply; scrolls when there are more than fit.
- **Menus** — **Game** (New Run / Quit), **View** (Deck/Relics, Map popups). `Alt+G` / `Alt+V`.

### How it maps onto the harness

The client is the harness's read/list/apply trio wired to the screen:

```
host = GameHost.StartNewRun(seed, new[] { character }, ascension);
host.EnterFirstRoom();
// each refresh:
state   = host.GetState();      // → coloured board (BoardRenderer)
options = host.ListOptions();   // → the Actions list
host.Apply(options[picked]);    // on Enter, then refresh
```

`StartNewRun(seed, IReadOnlyList<CharacterModel>, ascension)` is the harness overload added for
character selection.

### Logging

The game logs through `GD.Print`/`GD.PrintErr` (the GodotSharp shim). The shim's `GD.Out`/`GD.Err`
are redirectable; `Program.cs` points them at `sts2-tui.log` (next to the built exe) so the chatter
never touches the screen the Terminal.Gui driver owns. (Tests are unaffected — `GD` still defaults
to the live console.) Look in `sts2-tui.log` if something misbehaves.

### Self-test (no terminal)

A non-interactive mode (no Terminal.Gui) auto-plays random legal options, printing a one-line
summary per step — handy for verifying the render-model + harness path on a machine without a TTY:

```sh
dotnet run --project src/Sts2.Tui -- --smoke [seed] [steps] [ascension] [character]
```
