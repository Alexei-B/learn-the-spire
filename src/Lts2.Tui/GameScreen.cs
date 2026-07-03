using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Lts2.Agent;
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
    private readonly FrameView _optionsFrame;
    private readonly OptionsView _optionsView;
    private readonly CombatDecisionView _combatView;
    private readonly FrameView _logFrame;
    private readonly BoardView _log;
    private readonly Label _msg;

    // During card targeting in combat, the hotkey badges to overlay on the enemies (combat id → badge),
    // or null when not targeting. Threaded into the board renderer so the enemies show which key hits them.
    private IReadOnlyDictionary<uint, TargetBadge>? _targetOverlay;

    // Filled by the board renderer each draw with each targetable enemy's clickable rectangle (in board
    // coordinates), so a click on the combat board can be resolved to the enemy under it while aiming.
    private readonly Dictionary<uint, System.Drawing.Rectangle> _targetHitRegions = new();

    private readonly GameLog _gameLog = new();

    // The shared session log (the game's GD output sink). Each run tees its own RunLog onto this so
    // engine-side output lands in both the session log and the per-run log. See StartRunLog.
    private readonly System.IO.TextWriter _sessionLog;
    private RunLog? _runLog;

    private GameHost? _host;
    private GameState? _state;
    private List<GameOption> _options = new();
    private bool _busy;
    private bool _gameOverShown;

    // The pluggable decision engines the "auto-play" (Tab) shortcut can use, and which one is active.
    // Swapping engines lets the app drive its recommendations from a different strategy (rules, random,
    // or an external/learned policy) behind the shared IDecisionEngine seam — the same seam agent
    // training and evaluation use. Built once in the constructor; the active one is highlighted in the
    // Strategy menu. An external policy server (a ProcessDecisionEngine) is appended when configured
    // via the LTS2_AGENT_CMD environment variable (see BuildEngines).
    private readonly IDecisionEngine[] _engines;
    private MenuItem[] _strategyItems = Array.Empty<MenuItem>();
    private int _engineIndex;
    private IDecisionEngine Engine => _engines[_engineIndex];

    // Set when the current state is a multi-select card choice (choose N of M): such a choice can't be
    // resolved by a single flat option, so the decision list shows one "open the picker" entry and
    // activating it opens the interactive selector (see OpenCardPicker). Null otherwise.
    private PendingChoiceView? _multiChoice;

    // The map node the currently-highlighted move option leads to (null when the selection is not a
    // move). Read by the board/side renderers each draw so the map highlights where a move would take
    // you and dims the nodes that don't follow on from it.
    private Coord? _mapHighlight;

    public GameScreen(System.IO.TextWriter sessionLog)
    {
        _sessionLog = sessionLog;
        _engines = BuildEngines(sessionLog);
        _root = new Toplevel { ColorScheme = Theme.Base };

        _strategyItems = _engines.Select((e, i) =>
            new MenuItem($"_{i + 1}  {e.Name}", "", () => SelectEngine(i))
            {
                CheckType = MenuItemCheckStyle.Radio,
                Checked = i == _engineIndex,
            }).ToArray();

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
                new MenuBarItem("_Strategy", _strategyItems),
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
            // Only the decision area takes input; the display panes are non-interactive so Tab can't
            // move focus onto them (a focused-but-inert pane looks like the game has hung).
            CanFocus = false,
            TabStop = TabBehavior.NoStop,
        };
        _board = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _board.Clicked += OnBoardClick;
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
            CanFocus = false,
            TabStop = TabBehavior.NoStop,
        };
        _side = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _sideFrame.Add(_side);

        _optionsFrame = new FrameView
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
        // The combat hand/decision view, shown only in combat (see Refresh) where it replaces the list.
        _combatView = new CombatDecisionView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(), Visible = false };
        _combatView.Chosen += OnCombatChoice;
        _combatView.TargetingChanged += OnTargetingChanged;
        _optionsFrame.Add(_optionsView, _combatView);

        _logFrame = new FrameView
        {
            Title = "Log",
            X = Pos.Right(_optionsFrame),
            Y = Pos.Bottom(_boardFrame),
            Width = Dim.Fill(),
            Height = Dim.Fill(1),
            ColorScheme = Theme.Frame,
            BorderStyle = LineStyle.Rounded,
            CanFocus = false,
            TabStop = TabBehavior.NoStop,
        };
        _log = new BoardView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
        _log.SetRenderer((w, h) => _gameLog.Render(w, h));
        _logFrame.Add(_log);

        _msg = new Label { X = 1, Y = Pos.AnchorEnd(1), Width = Dim.Fill(1), Text = "" };

        _root.Add(menu, _boardFrame, _sideFrame, _optionsFrame, _logFrame, _msg);
        Refresh();
    }

    public Toplevel Root => _root;

    /// <summary>
    /// The decision engines available in the Strategy menu: the built-in rules and random engines, plus
    /// an external policy server when one is configured. Set <c>LTS2_AGENT_CMD</c> to the command that
    /// launches a decision server speaking the agent line protocol (e.g. <c>python</c>), with
    /// <c>LTS2_AGENT_ARGS</c> for its arguments (e.g. the script path) and <c>LTS2_AGENT_NAME</c> for the
    /// menu label. A launch failure is logged and skipped rather than blocking startup.
    /// </summary>
    private static IDecisionEngine[] BuildEngines(System.IO.TextWriter log)
    {
        var engines = new List<IDecisionEngine> { new RulesDecisionEngine(), new RandomDecisionEngine() };

        string? command = Environment.GetEnvironmentVariable("LTS2_AGENT_CMD");
        if (!string.IsNullOrWhiteSpace(command))
        {
            string? args = Environment.GetEnvironmentVariable("LTS2_AGENT_ARGS");
            string name = Environment.GetEnvironmentVariable("LTS2_AGENT_NAME") ?? "External";
            try
            {
                engines.Add(ProcessDecisionEngine.Launch(name, command, args, log: log.WriteLine));
            }
            catch (Exception ex)
            {
                log.WriteLine($"[agent] failed to launch external engine '{command}': {ex.Message}");
            }
        }

        return engines.ToArray();
    }

    /// <summary>Tear down any engines that own external resources (e.g. a policy subprocess). Called on
    /// exit so a launched decision server is killed with the app.</summary>
    public void Shutdown()
    {
        foreach (IDecisionEngine engine in _engines)
        {
            (engine as IDisposable)?.Dispose();
        }
    }

    /// <summary>Switch the active decision engine (the Strategy menu), tick its radio item, and repaint so
    /// the new engine's "auto-play" recommendation shows immediately.</summary>
    private void SelectEngine(int index)
    {
        _engineIndex = index;
        for (int i = 0; i < _strategyItems.Length; i++)
        {
            _strategyItems[i].Checked = i == index;
        }
        _msg.Text = $" Strategy engine: {Engine.Name}";
        Refresh();
    }

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
        StartRunLog(cfg.Seed, cfg.Character.Id.Entry, $"NEW RUN — ascension {cfg.Ascension}");
        Refresh();
        return true;
    }

    /// <summary>
    /// Begin a fresh per-run log file and route the game's GD output into it (teed with the session
    /// log). Called when a run starts or is loaded, so each run gets its own analysable log. Best-effort:
    /// if the log can't be opened, play continues without it.
    /// </summary>
    private void StartRunLog(string seed, string character, string how)
    {
        _runLog?.Dispose();
        _runLog = RunLog.Create(seed, character);
        if (_runLog is null)
        {
            Godot.GD.Out = _sessionLog;
            Godot.GD.Err = _sessionLog;
            return;
        }
        _runLog.Banner($"{how} — {character} · seed {seed}");
        var tee = new TeeTextWriter(_sessionLog, _runLog.Writer);
        Godot.GD.Out = tee;
        Godot.GD.Err = tee;
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
        // The board shows the enemies' target badges while the combat view is aiming a targeted card, and
        // records their clickable rectangles into _targetHitRegions so a board click resolves the target.
        _board.SetRenderer((w, h) => BoardRenderer.Board(state, w, h, _mapHighlight, _targetOverlay, _targetHitRegions));

        bool inCombat = state.Phase is GamePhase.Combat or GamePhase.Choice;
        _sideFrame.Title = inCombat ? "Piles" : "Map";
        _side.SetRenderer((w, h) => BoardRenderer.SidePanel(state, w, h, _mapHighlight));

        _options = _host.ListOptions().ToList();

        // The active decision engine's suggested move (the Tab "auto-play" pick) for this state, or null
        // if it has no opinion here (e.g. the rules engine off the battlefield). Combat draws it as a
        // "(tab)" marker on the hand card; other phases mark it in the options list.
        GameOption? recommended = Engine.Recommend(state, _options);

        // In active combat the decision area draws the hand as interactive card art (its own layout); every
        // other phase (including a mid-combat card Choice) uses the scrolling options list. Any aiming
        // overlay is cleared on refresh — the state has moved on.
        bool combatDecision = state.Phase == GamePhase.Combat && !state.IsGameOver;
        _targetOverlay = null;
        ApplyCombatLayout(state, combatDecision);

        if (combatDecision)
        {
            _multiChoice = null;
            _combatView.SetState(state, _options, recommended);
            _combatView.SetFocus();
        }
        else
        {
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
                // The engine's pick (if any) is Tab-selectable in the list too, marked "(tab)".
                int autoIdx = recommended is null ? -1 : _options.IndexOf(recommended);
                _optionsView.SetEntries(entries, endTurn >= 0 ? endTurn : null,
                    autoIndex: autoIdx >= 0 ? autoIdx : null);
            }
            _optionsView.SetFocus();
        }
        _log.SetNeedsDraw();

        _msg.Text = state.IsGameOver
            ? " Run over.  Game ▸ New Run to play again."
            : combatDecision
                ? " 1-9 play card · 0 end turn · Tab auto-play · targeted card → pick target 1-9 (Esc cancels) · Alt+G Game"
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

    /// <summary>
    /// Size and swap the decision area for the current phase. In combat the hand card view takes over the
    /// full width and a fixed height sized to the number of card rows the hand needs (see
    /// <see cref="CombatDecisionView.ContentHeight"/>); the board/piles shrink to whatever is left. Every
    /// other phase restores the normal split (list on the left, log on the right).
    /// </summary>
    private void ApplyCombatLayout(GameState state, bool combatDecision)
    {
        _combatView.Visible = combatDecision;
        _optionsView.Visible = !combatDecision;
        _logFrame.Visible = !combatDecision;

        if (combatDecision)
        {
            _optionsFrame.Title = "Hand";
            _optionsFrame.Width = Dim.Fill();
            int innerWidth = Math.Max(20, _root.Frame.Width - 2);
            PlayerCombatView? cs = state.Players.Count > 0 ? state.Players[0].CombatState : null;
            int handCount = cs?.Hand.Count ?? 0;
            int potionCount = state.Players.Count > 0 ? state.Players[0].Potions.Count(p => p is not null) : 0;
            int decisionsHeight = CombatDecisionView.ContentHeight(handCount, potionCount, innerWidth) + 2; // + frame borders
            _boardFrame.Height = Dim.Fill(decisionsHeight + 1);
            _sideFrame.Height = Dim.Fill(decisionsHeight + 1);
        }
        else
        {
            _optionsFrame.Title = "Decisions  (↑↓ · 0-9 · Enter)";
            _optionsFrame.Width = Dim.Percent(68);
            _boardFrame.Height = Dim.Percent(72);
            _sideFrame.Height = Dim.Percent(72);
        }
        _root.SetNeedsLayout();
    }

    /// <summary>Apply a card/target/end-turn option chosen in the combat hand view (pumps the game).</summary>
    private void OnCombatChoice(GameOption option)
    {
        if (_busy || _host is null)
        {
            return;
        }
        _targetOverlay = null;
        GameState? before = _state;
        string header = OptionHeader(option, before);
        Resolve(header, $" Applied: {option.Description}", before, host => host.Apply(option));
    }

    /// <summary>
    /// A click on the combat board. While aiming a targeted card, a left-click on a viable enemy plays the
    /// card at it and a right-click cancels aiming; otherwise clicks on the board are ignored.
    /// </summary>
    private void OnBoardClick(int col, int row, bool right)
    {
        if (_busy || !_combatView.IsTargeting)
        {
            return;
        }
        if (right)
        {
            _combatView.StopTargeting();
            return;
        }
        foreach (KeyValuePair<uint, System.Drawing.Rectangle> region in _targetHitRegions)
        {
            if (region.Value.Contains(col, row))
            {
                _combatView.SelectTargetByCombatId(region.Key);
                return;
            }
        }
    }

    /// <summary>The combat view started/stopped aiming a card: repaint the board with the target badges.</summary>
    private void OnTargetingChanged()
    {
        _targetOverlay = _combatView.TargetOverlay;
        _board.SetNeedsDraw();
        _msg.Text = _combatView.Prompt is { } p
            ? " " + p
            : " 1-9 play card · 0 end turn · Tab auto-play · targeted card → pick target 1-9 (Esc cancels) · Alt+G Game";
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
        // Record the action *before* it runs and flush it, so if the action hangs (e.g. an enemy turn
        // that never ends) the run log's last line names the action that got stuck.
        _runLog?.Action(header);
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
                _runLog?.Error(ex);
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
            IReadOnlyList<Line> events = _gameLog.Record(header, before, after);
            _runLog?.Result(after, events);
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
            GameState loaded = host.GetState();
            StartRunLog(
                loaded.Seed,
                loaded.Players.Count > 0 ? loaded.Players[0].Character : "?",
                $"LOADED — {System.IO.Path.GetFileName(path)} · {loaded.Phase} A{loaded.ActIndex + 1} F{loaded.Floor}");
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
