using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Sts2.Harness;
using Terminal.Gui;

namespace Sts2.Tui;

/// <summary>
/// The live game screen: the state board on top, and the decision area (the legal options, with
/// their localized descriptions) on the bottom. Reading state and listing options is synchronous;
/// applying an option *pumps the game* (a card play, or the whole enemy turn), which is run on a
/// background thread — never the Terminal.Gui UI thread, whose SynchronizationContext the game's
/// async continuations would capture and then deadlock against while the UI thread waits.
/// </summary>
internal sealed class GameScreen
{
    private readonly Toplevel _root;
    private readonly FrameView _boardFrame;
    private readonly BoardView _board;
    private readonly FrameView _sideFrame;
    private readonly BoardView _side;
    private readonly OptionsView _optionsView;
    private readonly Label _msg;

    private GameHost? _host;
    private GameState? _state;
    private List<GameOption> _options = new();
    private bool _busy;
    private bool _gameOverShown;

    public GameScreen()
    {
        _root = new Toplevel { ColorScheme = Theme.Base };

        var menu = new MenuBar
        {
            ColorScheme = Theme.Menu,
            Menus = new[]
            {
                new MenuBarItem("_Game", new[]
                {
                    new MenuItem("_New Run", "", () => PromptNewRun()),
                    new MenuItem("_Quit", "", () => Application.RequestStop(_root)),
                }),
                new MenuBarItem("_View", new[]
                {
                    new MenuItem("_Deck / Relics", "", ShowDeck),
                    new MenuItem("_Map", "", ShowMap),
                }),
            },
        };

        _boardFrame = new FrameView
        {
            Title = "Board",
            X = 0,
            Y = 1,
            Width = Dim.Percent(68),
            Height = Dim.Percent(72),
            ColorScheme = Theme.Frame,
            BorderStyle = LineStyle.Rounded,
        };
        _board = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _boardFrame.Add(_board);

        _sideFrame = new FrameView
        {
            Title = "Map",
            X = Pos.Right(_boardFrame),
            Y = 1,
            Width = Dim.Fill(),
            Height = Dim.Percent(72),
            ColorScheme = Theme.Frame,
            BorderStyle = LineStyle.Rounded,
        };
        _side = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _sideFrame.Add(_side);

        var optionsFrame = new FrameView
        {
            Title = "Decisions  (↑↓ · 1-9 · Enter)",
            X = 0,
            Y = Pos.Bottom(_boardFrame),
            Width = Dim.Fill(),
            Height = Dim.Fill(1),
            ColorScheme = Theme.Frame,
            BorderStyle = LineStyle.Rounded,
        };
        _optionsView = new OptionsView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _optionsView.Activated += Choose;
        optionsFrame.Add(_optionsView);

        _msg = new Label { X = 1, Y = Pos.AnchorEnd(1), Width = Dim.Fill(1), Text = "" };

        _root.Add(menu, _boardFrame, _sideFrame, optionsFrame, _msg);
        Refresh();
    }

    public Toplevel Root => _root;

    /// <summary>Show the new-run dialog and, if confirmed, start the run. Returns false if cancelled.</summary>
    public bool PromptNewRun()
    {
        RunConfig? cfg = NewRunDialog.Show();
        if (cfg is null)
        {
            return false;
        }
        _gameOverShown = false;
        _host = GameHost.StartNewRun(cfg.Seed, new[] { cfg.Character }, cfg.Ascension);
        _host.EnterFirstRoom();
        Refresh();
        return true;
    }

    /// <summary>Re-read the game state and repaint the board + decision area.</summary>
    public void Refresh()
    {
        if (_host is null)
        {
            _state = null;
            _boardFrame.Title = "Board";
            _board.SetLines(new List<Line> { new Line().Dim("No run in progress — Game ▸ New Run.") });
            _side.SetLines(new List<Line>());
            _optionsView.SetEntries(new List<OptionsView.Entry>());
            _options = new List<GameOption>();
            _msg.Text = " Game ▸ New Run to begin.";
            return;
        }

        GameState state = _host.GetState();
        _state = state;
        _boardFrame.Title = $"Board — {state.Phase}";
        _board.SetRenderer((w, h) => BoardRenderer.Board(state, w, h));

        bool inCombat = state.Phase is GamePhase.Combat or GamePhase.Choice;
        _sideFrame.Title = inCombat ? "Piles" : "Map";
        _side.SetRenderer((w, h) => BoardRenderer.SidePanel(state, w, h));

        _options = _host.ListOptions().ToList();
        var entries = new List<OptionsView.Entry>(_options.Count);
        foreach (GameOption o in _options)
        {
            entries.Add(new OptionsView.Entry(BoardRenderer.OptionLabel(o, state), BoardRenderer.OptionDescSegs(o, state)));
        }
        _optionsView.SetEntries(entries);
        _optionsView.SetFocus();

        _msg.Text = state.IsGameOver
            ? " Run over.  Game ▸ New Run to play again."
            : " ↑↓ select · 1-9 quick-pick · Enter apply · Alt+G Game · Alt+V View";

        if (state.IsGameOver && !_gameOverShown)
        {
            _gameOverShown = true;
            MessageBox.Query(
                state.IsVictory ? "Victory" : "Defeat",
                $"\nReached Act {state.ActIndex + 1}, floor {state.Floor}.\nFinal score: {state.Score}\n",
                "OK");
        }
    }

    private void Choose(int index)
    {
        if (_busy || _host is null || index < 0 || index >= _options.Count)
        {
            return;
        }
        GameHost host = _host;
        GameOption option = _options[index];
        _busy = true;
        _msg.Text = " Resolving…";

        // Pump the game off the UI thread (see the class remark on the deadlock — Terminal.Gui installs
        // a MainLoopSyncContext the game's async continuations would otherwise capture). A thread-pool
        // thread has no SynchronizationContext; clear it explicitly too, so continuations never post
        // back to the (blocked) UI thread. Marshal the result back to the UI thread to repaint.
        Task.Run(() =>
        {
            System.Threading.SynchronizationContext.SetSynchronizationContext(null);
            try
            {
                host.Apply(option);
                return (string?)null;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex);
                return ex.Message;
            }
        }).ContinueWith(t => Application.Invoke(() =>
        {
            _busy = false;
            _msg.Text = t.Result is string err ? $" Rejected: {err}" : $" Applied: {option.Description}";
            Refresh();
        }));
    }

    private void ShowDeck()
    {
        if (_host is null || _busy)
        {
            return;
        }
        PlayerState p = _host.GetState().Players[0];
        Popup("Deck / Relics", BoardRenderer.Deck(p));
    }

    private void ShowMap()
    {
        if (_host is null || _busy)
        {
            return;
        }
        MapView? map = _host.GetState().Map;
        if (map is null)
        {
            MessageBox.Query("Map", "\nNo map here (you're not on the overworld).\n", "OK");
            return;
        }
        var lines = new List<Line>();
        BoardRenderer.MapLines(map, lines, 60, 26);
        Popup($"Map — Act {map.ActIndex + 1}", lines);
    }

    private static void Popup(string title, List<Line> lines)
    {
        int width = Math.Min(Math.Max(lines.Count == 0 ? 40 : lines.Max(l => l.Sum(s => s.Text.Length)) + 4, 40), 100);
        int height = Math.Min(lines.Count + 4, 30);

        var dlg = new Dialog { Title = title, Width = width, Height = height, ColorScheme = Theme.Base };
        var view = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(1) };
        view.SetLines(lines);
        var ok = new Button { Text = "Close", IsDefault = true };
        ok.Accepting += (_, e) => { e.Cancel = true; Application.RequestStop(dlg); };
        dlg.Add(view);
        dlg.AddButton(ok);
        Application.Run(dlg);
        dlg.Dispose();
    }
}
