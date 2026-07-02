using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Lts2.Harness;
using Terminal.Gui;

namespace Lts2.Tui;

/// <summary>
/// The live game screen: the state board (top-left), the map/piles side panel (top-right), the
/// decision area (bottom-left) and a scrolling event log (bottom-right). Reading state and listing
/// options is synchronous; applying an option *pumps the game* (a card play, or the whole enemy
/// turn), which is run on a background thread — never the Terminal.Gui UI thread, whose
/// SynchronizationContext the game's async continuations would capture and then deadlock against
/// while the UI thread waits. Loading a save pumps the game too, so it runs off-thread as well.
/// </summary>
internal sealed class GameScreen
{
    private readonly Toplevel _root;
    private readonly FrameView _boardFrame;
    private readonly BoardView _board;
    private readonly FrameView _sideFrame;
    private readonly BoardView _side;
    private readonly OptionsView _optionsView;
    private readonly FrameView _logFrame;
    private readonly BoardView _log;
    private readonly Label _msg;

    private readonly GameLog _gameLog = new();

    private GameHost? _host;
    private GameState? _state;
    private List<GameOption> _options = new();
    private bool _busy;
    private bool _gameOverShown;

    // Set when the current state is a multi-select card choice (choose N of M): such a choice can't be
    // resolved by a single flat option, so the decision list shows one "open the picker" entry and
    // activating it opens the interactive selector (see OpenCardPicker). Null otherwise.
    private PendingChoiceView? _multiChoice;

    // The map node the currently-highlighted move option leads to (null when the selection is not a
    // move). Read by the board/side renderers each draw so the map highlights where a move would take
    // you and dims the nodes that don't follow on from it.
    private Coord? _mapHighlight;

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
                    new MenuItem("_Continue (autosave)", "", ContinueRun),
                    new MenuItem("_Save Run", "", SaveRun),
                    new MenuItem("_Load Run", "", LoadRun),
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
            Title = "Decisions  (↑↓ · 0-9 · Enter)",
            X = 0,
            Y = Pos.Bottom(_boardFrame),
            Width = Dim.Percent(68),
            Height = Dim.Fill(1),
            ColorScheme = Theme.Frame,
            BorderStyle = LineStyle.Rounded,
        };
        _optionsView = new OptionsView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _optionsView.Activated += Choose;
        _optionsView.SelectionChanged += OnSelectionChanged;
        optionsFrame.Add(_optionsView);

        _logFrame = new FrameView
        {
            Title = "Log",
            X = Pos.Right(optionsFrame),
            Y = Pos.Bottom(_boardFrame),
            Width = Dim.Fill(),
            Height = Dim.Fill(1),
            ColorScheme = Theme.Frame,
            BorderStyle = LineStyle.Rounded,
        };
        _log = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _log.SetRenderer((w, h) => _gameLog.Render(w, h));
        _logFrame.Add(_log);

        _msg = new Label { X = 1, Y = Pos.AnchorEnd(1), Width = Dim.Fill(1), Text = "" };

        _root.Add(menu, _boardFrame, _sideFrame, optionsFrame, _logFrame, _msg);
        Refresh();
    }

    public Toplevel Root => _root;

    /// <summary>
    /// Show the opening dialog and act on it: start a new run, resume the autosave, or (cancelled)
    /// return false so the caller can exit. A resume loads asynchronously (see <see cref="LoadFrom"/>).
    /// </summary>
    public bool PromptNewRun()
    {
        NewRunChoice? choice = NewRunDialog.Show();
        if (choice is null)
        {
            return false;
        }
        if (choice.Continue)
        {
            LoadFrom(SaveStore.AutosavePath);
            return true;
        }

        RunConfig cfg = choice.Config!;
        _gameOverShown = false;
        _host = GameHost.StartNewRun(cfg.Seed, new[] { cfg.Character }, cfg.Ascension);
        _host.EnterFirstRoom();
        _gameLog.Clear();
        _gameLog.Note($"New run — {cfg.Character.Id.Entry}, ascension {cfg.Ascension}, seed {cfg.Seed}.");
        Refresh();
        return true;
    }

    /// <summary>Re-read the game state and repaint the board, side panel, decisions and log.</summary>
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
        _mapHighlight = null;
        _boardFrame.Title = $"Board — {state.Phase}";
        _board.SetRenderer((w, h) => BoardRenderer.Board(state, w, h, _mapHighlight));

        bool inCombat = state.Phase is GamePhase.Combat or GamePhase.Choice;
        _sideFrame.Title = inCombat ? "Piles" : "Map";
        _side.SetRenderer((w, h) => BoardRenderer.SidePanel(state, w, h, _mapHighlight));

        _options = _host.ListOptions().ToList();

        // A multi-select card choice (choose N of M) surfaces as a single "open the picker" entry —
        // its cards are toggled and confirmed in the interactive selector, not applied one at a time.
        _multiChoice = state.Phase == GamePhase.Choice && state.PendingChoice is { MaxSelect: > 1 } mc ? mc : null;
        if (_multiChoice is { } choice)
        {
            string range = choice.MinSelect == choice.MaxSelect ? $"{choice.MaxSelect}" : $"{choice.MinSelect}-{choice.MaxSelect}";
            string verb = choice.IsUpgradeSelection ? "forge" : "select";
            var label = new List<Seg> { new($"Choose {range} card(s) to {verb}…", Theme.Teal) };
            var desc = new List<Seg> { new("Open the picker — Space toggles each card, Enter confirms.", Theme.Dim) };
            _optionsView.SetEntries(new List<OptionsView.Entry> { new(label, desc) });
        }
        else
        {
            var entries = new List<OptionsView.Entry>(_options.Count);
            foreach (GameOption o in _options)
            {
                entries.Add(new OptionsView.Entry(BoardRenderer.OptionLabel(o, state), BoardRenderer.OptionDescSegs(o, state)));
            }
            int endTurn = _options.FindIndex(o => o.Kind == OptionKind.EndTurn);
            // In combat, Tab applies the default-strategy pick; surface which option that is so the list
            // can mark it "(tab)" and Tab can activate it (see OptionsView).
            int? autoIndex = null;
            if (state.Phase == GamePhase.Combat
                && CombatStrategy.ChooseDefaultMove(state, _options) is { } autoPick)
            {
                int i = _options.IndexOf(autoPick);
                autoIndex = i >= 0 ? i : null;
            }
            _optionsView.SetEntries(entries, endTurn >= 0 ? endTurn : null, autoIndex: autoIndex);
        }
        _optionsView.SetFocus();
        _log.SetNeedsDraw();

        bool combatPhase = state.Phase is GamePhase.Combat;
        _msg.Text = state.IsGameOver
            ? " Run over.  Game ▸ New Run to play again."
            : combatPhase
                ? " ↑↓ select · 0-9 quick-pick (0=end turn) · Enter apply · Tab auto-play · Alt+G Game · Alt+V View"
                : " ↑↓ select · 0-9 quick-pick (0=end turn) · Enter apply · Alt+G Game · Alt+V View";

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
        if (_busy || _host is null)
        {
            return;
        }
        // A multi-select card choice opens the interactive picker instead of applying a flat option.
        if (_multiChoice is { } choice)
        {
            OpenCardPicker(choice);
            return;
        }
        if (index < 0 || index >= _options.Count)
        {
            return;
        }
        GameOption option = _options[index];
        GameState? before = _state;
        string header = OptionHeader(option, before);
        Resolve(header, $" Applied: {option.Description}", before, host => host.Apply(option));
    }

    /// <summary>
    /// Open the modal card picker for a multi-select choice; on confirm, resolve it with the chosen
    /// cards. Cancelling leaves the choice pending (the "open the picker" option stays available).
    /// </summary>
    private void OpenCardPicker(PendingChoiceView choice)
    {
        IReadOnlyList<int>? picked = CardSelectDialog.Show(choice);
        if (picked is null || _host is null)
        {
            return;
        }
        GameState? before = _state;
        string names = string.Join(", ", picked.Select(i => BoardRenderer.CardDisplayName(choice.Options[i])));
        string header = picked.Count == 0 ? "Skip selection" : $"Select {names}";
        Resolve(header, $" Applied: {header}", before, host => host.ApplyCardChoice(picked));
    }

    /// <summary>
    /// Pump a game action off the UI thread, then repaint. Applying an option (or resolving a choice)
    /// runs the game's async continuations, which must not run on the Terminal.Gui UI thread (see the
    /// class remark — its MainLoopSyncContext would be captured and deadlock). A thread-pool thread has
    /// no SynchronizationContext; clear it explicitly too. The result is marshalled back to repaint.
    /// </summary>
    private void Resolve(string header, string appliedMsg, GameState? before, Action<GameHost> apply)
    {
        GameHost host = _host!;
        _busy = true;
        _msg.Text = " Resolving…";
        Task.Run(() =>
        {
            SynchronizationContext.SetSynchronizationContext(null);
            try
            {
                apply(host);
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
            if (t.Result is string err)
            {
                _msg.Text = $" Rejected: {err}";
                Refresh();
                return;
            }

            GameState after = host.GetState();
            _gameLog.Record(header, before, after);
            // Checkpoint between rooms (the harness snapshot is only valid out of combat).
            if (after.Phase == GamePhase.Map && !after.IsGameOver)
            {
                SaveStore.Autosave(host, after);
            }
            _msg.Text = appliedMsg;
            Refresh();
        }));
    }

    /// <summary>
    /// When the highlighted option is a map move, mark its destination so the map view highlights that
    /// node and dims the ones that don't follow on from it. Any other selection clears the highlight.
    /// </summary>
    private void OnSelectionChanged(int index)
    {
        Coord? highlight = index >= 0 && index < _options.Count && _options[index].Kind == OptionKind.MoveTo
            ? _options[index].Coord
            : null;
        if (Nullable.Equals(highlight, _mapHighlight))
        {
            return;
        }
        _mapHighlight = highlight;
        _board.SetNeedsDraw();
        _side.SetNeedsDraw();
    }

    /// <summary>A short, human-readable header for the log describing the option just applied.</summary>
    private static string OptionHeader(GameOption option, GameState? state)
    {
        if (state is null)
        {
            return option.Description;
        }
        string label = string.Concat(BoardRenderer.OptionLabel(option, state).Select(s => s.Text)).Trim();
        return string.IsNullOrWhiteSpace(label) ? option.Description : label;
    }

    // ---- Save / load -----------------------------------------------------------

    private void SaveRun()
    {
        if (_host is null || _busy)
        {
            return;
        }
        GameState s = _host.GetState();
        if (s.Phase is GamePhase.Combat or GamePhase.Choice)
        {
            MessageBox.Query("Save Run", "\nYou can't save during combat.\nSave from the map, between rooms.\n", "OK");
            return;
        }
        if (s.IsGameOver)
        {
            MessageBox.Query("Save Run", "\nThe run is over — nothing to save.\n", "OK");
            return;
        }
        string? name = SaveDialogs.PromptSaveName($"{s.Players[0].Character}-{s.Seed}");
        if (name is null)
        {
            return;
        }
        try
        {
            SaveStore.Save(_host, s, name);
            _msg.Text = " Run saved.";
        }
        catch (Exception ex)
        {
            MessageBox.ErrorQuery("Save failed", "\n" + ex.Message + "\n", "OK");
        }
    }

    private void LoadRun()
    {
        if (_busy)
        {
            return;
        }
        string? path = SaveDialogs.PromptLoad();
        if (path is not null)
        {
            LoadFrom(path);
        }
    }

    private void ContinueRun()
    {
        if (_busy)
        {
            return;
        }
        if (!SaveStore.HasAutosave)
        {
            MessageBox.Query("Continue", "\nNo autosave found.\n", "OK");
            return;
        }
        LoadFrom(SaveStore.AutosavePath);
    }

    /// <summary>Restore a run from a save file off the UI thread (restore pumps the game), then repaint.</summary>
    private void LoadFrom(string path)
    {
        if (_busy)
        {
            return;
        }
        _busy = true;
        _msg.Text = " Loading…";
        Task.Run(() =>
        {
            SynchronizationContext.SetSynchronizationContext(null);
            try
            {
                return ((GameHost?)SaveStore.Load(path), (string?)null);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex);
                return ((GameHost?)null, ex.Message);
            }
        }).ContinueWith(t => Application.Invoke(() =>
        {
            _busy = false;
            (GameHost? host, string? err) = t.Result;
            if (host is null)
            {
                MessageBox.ErrorQuery("Load failed", "\n" + err + "\n", "OK");
                _msg.Text = " Load failed.";
                return;
            }
            _host = host;
            _gameOverShown = false;
            _gameLog.Clear();
            _gameLog.Note("Loaded run.");
            Refresh();
            _msg.Text = " Run loaded.";
        }));
    }

    // ---- Popups ----------------------------------------------------------------

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
