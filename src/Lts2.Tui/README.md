# Lts2.Tui — full-screen terminal client for the headless harness

A **full-screen, terminal-owning** client (think ncurses / Dwarf Fortress) that plays single-player
**Slay the Spire 2** runs, driving the `Lts2.Harness` library (the real game logic, no Godot). It
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
dotnet run --project src/Lts2.Tui
```

## Localized names & descriptions (optional but recommended)

The board, the action list, and a **Details** panel show real card/relic/potion/event names and
descriptions (with the actual numbers — "Deal 8 damage. Apply 2 Vulnerable.") via the
**`Lts2.Localization`** library. That text lives only in the game `.pck`, so extract it once per
clone (it lands under the gitignored `lib/`):

```sh
pwsh scripts/extract-localization.ps1   # uses GDRE Tools to pull res://localization/eng/*.json
```

If you skip this, the TUI still works — it just shows model ids (e.g. `BASH`) instead of names.
The harness itself never depends on this; only the TUI (via the library) does.

A **New Run** dialog opens first (character, ascension 0–10, seed; plus **Continue** when an autosave
exists). Then the main screen:

```
 Game   View
┌ Board — Combat ──────────────────────────────────┐┌ Piles ─────────────┐
│ Act 1 · Floor 2 · 80/80 HP · 99g …               ││ DRAW (5)           │
│ ┌ IRONCLAD ─────────┐  ┌ #1 Leaf Slime (S) ────┐ ││   Strike x4  Bash  │
│ │ ███████████ 80/80 │  │ ██████████ 12/12      │ ││ DISCARD (0)        │
│ │ Energy ●●● Hand 5 │  │ Intent: Attack 3      │ ││ EXHAUST (0)        │
│ └───────────────────┘  └───────────────────────┘ ││                    │
│ …                                                 ││                    │
└───────────────────────────────────────────────────┘└────────────────────┘
┌ Decisions  (↑↓ · 0-9 · Enter) ─────────────┐┌ Log ────────────────────┐
│ [1] ● Strike → #1                          ││ ▸ End turn              │
│       Deal 6 damage.                       ││  · IRONCLAD took 7 dmg  │
│ [2] ● Defend                               ││  · gained Vulnerable 2  │
│       Gain 5 Block.                        ││ ▸ ● Strike → #1         │
└─────────────────────────────────────────────┘└─────────────────────────┘
```

- **Board** (top-left) — the live state for the phase. In **combat**: allies (players + Osty) on the
  left and enemies on the right, each a bordered box with a coloured **health bar** (red current HP,
  dark lost HP, green = HP poison will remove this turn, purple = doom threshold; the border turns
  green/purple if they'd die to poison/doom this turn, grey if they have block), an info line
  (energy as teal ●/○ circles + hand / enemy intent), and its powers. On the **map** screen: the act
  map with connections drawn between rooms — the header names the act's **boss** and the boss room
  shows as a `B` node at the top. In an **event**: the flavour body text plus each option's outcome
  text, with the real per-run numbers (energy shown as teal circles, colours applied).
- **Side panel** (top-right) — the act **map** (with connections) on every screen, except in combat
  where it shows your **draw / discard / exhaust piles**. Press **`r`** (or **View ▸ Agent Ranking**) to
  swap it for the **agent ranking panel** (see below).
- **Agent ranking panel** (toggle with **`r`**) — the active **Strategy** engine's full scored ranking for
  the current decision, sorted by score (what `Tab` auto-play would apply is marked **▸**), each with its
  score and the engine's rationale. The panel title names the engine. It updates as the game advances and
  says so explicitly when the engine **declined** (no recommendation) or is still **evaluating** (an
  external agent can take a moment). The ranking is fetched once per decision point, off the UI thread, and
  shared with `Tab` — so opening the panel costs no extra agent round-trip and never blocks the UI.
- **Decisions** (bottom-left) — the legal options with their localized descriptions inline. Move with
  **↑/↓**, **0–9** to quick-pick (**0** is always *End turn* in combat), **Enter** to apply; scrolls
  when there are more than fit. Potions can also be used/discarded on the **rewards** screen. While a
  **map move** is highlighted, the map lights up that room and everything it can still reach, and
  dims the rooms that move can't lead to (`{x}` marks the selected room).
- **Log** (bottom-right) — a scrolling record of what changed on each decision (damage taken, cards
  gained or moved between piles, relics/potions/gold/powers, enemy defeats, phase changes), derived
  by diffing the state before/after each apply.
- **Menus** — **Game** (New Run / Continue / Save Run / Load Run / Quit), **View** (Deck/Relics, Map
  popups, Agent Ranking — the `r` panel), **Strategy** (pick the auto-play engine). `Alt+G` / `Alt+V` /
  `Alt+S`. Saving is only possible out of combat (on the map); the app also
  autosaves whenever you reach the map, so **Continue** resumes your latest checkpoint.

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
are redirectable; `Program.cs` points them at `lts2-tui.log` (next to the built exe) so the chatter
never touches the screen the Terminal.Gui driver owns. (Tests are unaffected — `GD` still defaults
to the live console.) Look in `lts2-tui.log` if something misbehaves.

### Self-test (no terminal)

Non-interactive modes (no Terminal.Gui) exercise the render-model, localization and harness path on a
machine without a TTY:

```sh
# Auto-play random legal options, one summary line per step, then print the event-log tail.
dotnet run --project src/Lts2.Tui -- --smoke [seed] [steps] [ascension] [character]

# Render the board / side panel / decisions of the first combat and map frame as plain text.
dotnet run --project src/Lts2.Tui -- --dump [seed] [character]

# Enter one event by type name (e.g. SelfHelpBook) and dump its body + per-option outcome text.
dotnet run --project src/Lts2.Tui -- --dumpevent [EventTypeName]
```
