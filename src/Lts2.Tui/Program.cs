using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Lts2.Localization;
using Terminal.Gui;

namespace Lts2.Tui;

internal static class Program
{
    private static int Main(string[] args)
    {
        // The game's logging (GD.Print/PrintErr in the shim) is routed to a file, leaving the real
        // console free for Terminal.Gui's screen driver. We do NOT redirect Console.Out globally,
        // because the driver writes the UI through it.
        string logPath = Path.Combine(AppContext.BaseDirectory, "lts2-tui.log");
        var logWriter = new StreamWriter(logPath, append: false) { AutoFlush = true };

        try
        {
            if (args.Length > 0 && args[0] == "--loctest")
            {
                Godot.GD.Out = logWriter;
                Godot.GD.Err = logWriter;
                return LocTest();
            }

            if (args.Length > 0 && args[0] == "--dump")
            {
                Godot.GD.Out = logWriter;
                Godot.GD.Err = logWriter;
                return Dump(args.Skip(1).ToArray());
            }

            if (args.Length > 0 && args[0] == "--dumpevent")
            {
                Godot.GD.Out = logWriter;
                Godot.GD.Err = logWriter;
                return DumpEvent(args.Skip(1).ToArray());
            }

            if (args.Length > 0 && args[0] == "--smoke")
            {
                Godot.GD.Out = logWriter;
                Godot.GD.Err = logWriter;
                return Smoke(args.Skip(1).ToArray());
            }
            return RunTui(logWriter);
        }
        finally
        {
            logWriter.Flush();
        }
    }

    // NOTE: keep direct references to game types (MegaCrit.*, Godot.*, Lts2.Localization.*) out of
    // Main's own body. Main is JIT-compiled and run at startup; a game-type reference there loads the
    // GodotSharp shim before Application.Init(), which makes Terminal.Gui's type scan crash (see the
    // ordering note in RunTui). Put such code in a separate method, called from Main.
    private static int LocTest()
    {
        GameRuntime.EnsureInitialized();
        Console.WriteLine($"Localizer.Available = {Localizer.Available}");
        foreach (string id in new[] { "BASH", "STRIKE_IRONCLAD", "ANGER", "BATTLE_TRANCE" })
        {
            Console.WriteLine($"CARD {id}: '{Localizer.CardName(id)}' :: {Localizer.CardDescription(id).Replace("\n", " / ")}");
        }
        foreach (string id in ModelDb.AllRelics.Take(3).Select(r => r.Id.Entry))
        {
            Console.WriteLine($"RELIC {id}: '{Localizer.RelicName(id)}' :: {Localizer.RelicDescription(id).Replace("\n", " / ")}");
        }
        foreach (string id in new[] { "ADRENALINE", "BASH" })
        {
            string raw = Localizer.CardDescription(id);
            string segs = string.Join(" | ", Markup.Parse(raw, Theme.Fg).Select(s => $"'{s.Text.Replace("\n", "\\n")}'@{ColorTag(s.Fg)}"));
            Console.WriteLine($"PARSE {id}: {segs}");
        }
        Console.WriteLine("EVENT (regular) ABYSSAL_BATHS/ABSTAIN title: " +
            (Localizer.EventOptionTitle("ABYSSAL_BATHS", "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN") ?? "<null>"));
        Console.WriteLine("EVENT (ancient) NEOW/FISHING_ROD title: " +
            (Localizer.EventOptionTitle("NEOW", "NEOW.pages.INITIAL.options.FISHING_ROD")?.ToString() ?? "<null → falls back to relic name>"));
        return 0;
    }

    // Render frames to plain text (colours dropped) for eyeballing layout without a terminal.
    private static int Dump(string[] args)
    {
        string seed = args.Length > 0 ? args[0] : "DUMP1";
        string? charName = args.Length > 1 ? args[1] : null;
        var rng = new Random(7);
        GameRuntime.EnsureInitialized();
        CharacterModel character = charName is null
            ? ModelDb.AllCharacters.First()
            : ModelDb.AllCharacters.First(c => c.Id.Entry.Contains(charName, StringComparison.OrdinalIgnoreCase));
        GameHost host = GameHost.StartNewRun(seed, new[] { character }, 0);
        host.EnterFirstRoom();

        bool dumpedCombat = false, dumpedMap = false;
        for (int i = 0; i < 200 && !(dumpedCombat && dumpedMap); i++)
        {
            GameState state = host.GetState();
            IReadOnlyList<GameOption> opts = host.ListOptions();
            if (state.Phase is GamePhase.Combat && !dumpedCombat)
            {
                DumpFrame("COMBAT", state, opts);
                dumpedCombat = true;
            }
            if (state.Phase is GamePhase.Map && !dumpedMap)
            {
                DumpFrame("MAP", state, opts);
                dumpedMap = true;
            }
            if (state.IsGameOver)
            {
                break;
            }
            if (opts.Count == 0)
            {
                break;
            }
            var pool = opts.Where(o => o.Kind != OptionKind.EndTurn).ToList();
            if (pool.Count == 0)
            {
                pool = opts.ToList();
            }
            host.Apply(pool[rng.Next(pool.Count)]);
        }
        return 0;
    }

    // Enter one event by type name (e.g. SelfHelpBook) and dump its board (body text) + decisions
    // (per-option outcome text), so the event-text rendering can be eyeballed without a terminal.
    private static int DumpEvent(string[] args)
    {
        string typeName = args.Length > 0 ? args[0] : "SelfHelpBook";
        GameRuntime.EnsureInitialized();
        Console.WriteLine($"Localizer.Available = {Localizer.Available}");
        CharacterModel character = ModelDb.AllCharacters.First();
        GameHost host = GameHost.StartNewRun("EVTDUMP", new[] { character }, 0);
        host.EnterFirstRoom();

        EventModel ev = ModelDb.ActsByIndex[0]
            .SelectMany(a => a.AllEvents)
            .Concat(ModelDb.AllSharedEvents)
            .First(e => e.GetType().Name == typeName);
        host.EnterEventDebug(ev);

        GameState state = host.GetState();
        DumpFrame($"EVENT {typeName}", state, host.ListOptions());
        return 0;
    }

    private static void DumpFrame(string tag, GameState state, IReadOnlyList<GameOption> opts)
    {
        Console.WriteLine($"================ {tag} (phase {state.Phase}) ================");
        Console.WriteLine("--- BOARD (w=90 h=30) ---");
        foreach (Line l in BoardRenderer.Board(state, 90, 30))
        {
            Console.WriteLine(string.Concat(l.Select(s => s.Text)));
        }
        Console.WriteLine("--- SIDE (w=34 h=30) ---");
        foreach (Line l in BoardRenderer.SidePanel(state, 34, 30))
        {
            Console.WriteLine(string.Concat(l.Select(s => s.Text)));
        }
        Console.WriteLine("--- DECISIONS ---");
        for (int i = 0; i < opts.Count; i++)
        {
            string label = string.Concat(BoardRenderer.OptionLabel(opts[i], state).Select(s => s.Text));
            Console.WriteLine($"[{i + 1}] {label}");
            string desc = string.Concat(BoardRenderer.OptionDescSegs(opts[i], state).Select(s => s.Text));
            if (desc.Length > 0)
            {
                Console.WriteLine($"      {desc.Replace("\n", " / ")}");
            }
        }
        Console.WriteLine();
    }

    private static string ColorTag(Terminal.Gui.Color c) =>
        c == Theme.Gold ? "gold" : c == Theme.Blue ? "blue" : c == Theme.Red ? "red" :
        c == Theme.Green ? "green" : c == Theme.Magenta ? "purple" : c == Theme.Teal ? "teal" :
        c == Theme.Orange ? "orange" : c == Theme.Pink ? "pink" : c == Theme.Fg ? "fg" : "?";

    private static int RunTui(StreamWriter logWriter)
    {
        // IMPORTANT ordering: Terminal.Gui's ConfigurationManager scans every *already-loaded*
        // assembly's types at Init (to gather config/theme defaults). Enumerating our intentionally-
        // incomplete GodotSharp shim / sts2 (their Godot source-generator types — MethodName/
        // PropertyName/SignalName/EventType — are not all defined, by design) throws
        // ReflectionTypeLoadException. So initialise Terminal.Gui FIRST, while the game assemblies are
        // not yet loaded, then route logging and boot the game (which loads them) afterwards — the
        // one-time scan has already run and won't see them.
        Application.Init();
        try
        {
            Theme.Init();

            Godot.GD.Out = logWriter;
            Godot.GD.Err = logWriter;
            GameRuntime.EnsureInitialized();

            var screen = new GameScreen(logWriter);

            // Open the first new-run dialog once the main loop is running (not before — running a
            // modal as the very first top-level corrupts the run stack). Cancelling it exits.
            Application.Invoke(() =>
            {
                if (!screen.PromptNewRun())
                {
                    Application.RequestStop(screen.Root);
                }
            });

            Application.Run(screen.Root);
            screen.Shutdown();
            screen.Root.Dispose();
            return 0;
        }
        finally
        {
            Application.Shutdown();
        }
    }

    /// <summary>
    /// Non-interactive self-test (no Terminal.Gui): start a run and auto-play random legal options,
    /// printing a one-line summary per step to stdout. Verifies the render-model + harness path.
    /// Usage: <c>--smoke [seed] [steps] [ascension] [character]</c>.
    /// </summary>
    private static int Smoke(string[] args)
    {
        string seed = args.Length > 0 ? args[0] : "SMOKE123";
        int steps = args.Length > 1 && int.TryParse(args[1], out int s) ? s : 60;
        int ascension = args.Length > 2 && int.TryParse(args[2], out int a) ? a : 0;
        string? charName = args.Length > 3 ? args[3] : null;
        var rng = new Random(12345);

        GameRuntime.EnsureInitialized();
        CharacterModel character = charName is null
            ? ModelDb.AllCharacters.First()
            : ModelDb.AllCharacters.First(c => c.Id.Entry.Contains(charName, StringComparison.OrdinalIgnoreCase));

        GameHost host = GameHost.StartNewRun(seed, new[] { character }, ascension);
        host.EnterFirstRoom();

        var gameLog = new GameLog();
        for (int i = 0; i < steps; i++)
        {
            GameState state = host.GetState();
            PlayerState p = state.Players[0];
            // Exercise the real render + localization path (no Terminal.Gui) so it can't hide bugs.
            int boardLines = BoardRenderer.Board(state, 100).Count;
            _ = BoardRenderer.SidePanel(state, 30);
            IReadOnlyList<GameOption> opts = host.ListOptions();
            foreach (GameOption op in opts)
            {
                _ = BoardRenderer.OptionLabel(op, state);
                _ = BoardRenderer.OptionDescSegs(op, state);
            }
            Console.WriteLine($"[{i,3}] {state.Phase,-12} A{state.ActIndex + 1} F{state.Floor} " +
                              $"HP {p.CurrentHp}/{p.MaxHp} {p.Gold}g  score {state.Score}  ({boardLines} board lines, {opts.Count} opts)");
            if (state.IsGameOver)
            {
                Console.WriteLine($"Run ended — victory={state.IsVictory}, score={state.Score}");
                break;
            }

            if (opts.Count == 0)
            {
                Console.WriteLine("No options; stopping.");
                break;
            }
            List<GameOption> nonEnd = opts.Where(o => o.Kind != OptionKind.EndTurn).ToList();
            List<GameOption> pool = nonEnd.Count > 0 ? nonEnd : opts.ToList();
            GameOption pick = pool[rng.Next(pool.Count)];
            Console.WriteLine($"      -> {pick.Description}");
            host.Apply(pick);
            // Exercise the event-log differ over the real state transition (before → after).
            gameLog.Record(string.Concat(BoardRenderer.OptionLabel(pick, state).Select(s => s.Text)), state, host.GetState());
        }

        Console.WriteLine();
        Console.WriteLine("---- event log tail ----");
        foreach (Line l in gameLog.Render(80, 24))
        {
            Console.WriteLine(string.Concat(l.Select(s => s.Text)));
        }
        Console.WriteLine("Smoke run completed without errors.");
        return 0;
    }
}
