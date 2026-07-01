using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace Lts2.Harness;

/// <summary>
/// Owns a single headless Slay the Spire 2 run: boots the game's logic singletons
/// without the Godot UI, then drives them deterministically from a seed.
/// </summary>
public sealed class GameHost
{
    /// <summary>The live run state. Treat as read-only from outside the harness.</summary>
    public RunState Run { get; private init; } = null!;

    /// <summary>The seed this run was created with.</summary>
    public string Seed { get; private init; } = "";

    /// <summary>
    /// The harness-controlled card selector installed as the game's <c>CardSelectCmd.Selector</c>.
    /// Surfaces mid-effect card choices so they resolve through <see cref="ListOptions(ulong)"/>
    /// / <see cref="Apply"/> instead of a UI screen.
    /// </summary>
    internal HarnessCardSelector Selector { get; private init; } = null!;

    // The CardSelectCmd selector stack is process-wide; track our scope so a new run can
    // replace the previous run's selector cleanly.
    private static IDisposable? _selectorScope;

    /// <summary>
    /// The rewards set currently on offer (for its <c>set.Player</c>), or null when not on the rewards
    /// screen. Surfaced as <see cref="GamePhase.Reward"/> and resolved through the reward options.
    /// </summary>
    internal MegaCrit.Sts2.Core.Rewards.RewardsSet? PendingRewards { get; private set; }

    // Additional players' post-combat reward sets awaiting their turn. A multi-player fight gives each
    // alive player their own rewards; the harness surfaces them one player at a time (PendingRewards),
    // dequeuing the next when the current player proceeds. Empty in single-player.
    private readonly System.Collections.Generic.Queue<MegaCrit.Sts2.Core.Rewards.RewardsSet> _queuedRewardSets = new();

    // The combat room whose rewards we have already generated, so we offer them exactly once
    // per fight even while the player lingers on the rewards screen before moving on.
    private MegaCrit.Sts2.Core.Rooms.AbstractRoom? _rewardedRoom;

    // The treasure room whose chest we have already opened, so the gold/extra-reward flow runs
    // exactly once per room (the relic picking that follows is driven by the agent's options).
    private MegaCrit.Sts2.Core.Rooms.TreasureRoom? _openedTreasureRoom;

    // A fire-and-forget room/effect task (e.g. a treasure room's DoExtraRewardsIfNeeded, or a
    // relic's on-obtain AfterObtained) kept in flight when it suspends on a custom reward screen so
    // ProceedFromRewards can resume it. Null otherwise.
    private System.Threading.Tasks.Task? _suspendedRoomTask;

    // A rest-site option resolution (ChooseLocalOption) task, kept in flight when it suspends on a
    // mid-effect card choice (e.g. Smith's upgrade selection) so Apply can resume it. Null otherwise.
    private System.Threading.Tasks.Task? _restChoiceTask;

    // A shop card-removal purchase task, kept in flight while it suspends on the deck card choice it
    // raises (the removal picks a card to remove), so Apply(SelectCards) can resume it. Null otherwise.
    // The matching entry is held so the harness can mark it used on success (the UI node that would
    // normally do that is null headless).
    private System.Threading.Tasks.Task<bool>? _shopRemovalTask;
    private MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardRemovalEntry? _shopRemovalEntry;

    // Custom rewards offered mid-effect by a relic/event (RewardsCmd.OfferCustom, e.g.
    // Kaleidoscope's two bonus card rewards). Unlike a post-combat set, the offering effect's
    // task is *suspended* inside RewardsSet.Offer until the agent takes/skips and proceeds, so
    // the agent makes the same explicit take-or-skip choice rather than the rewards being
    // silently auto-taken (which is what RewardsSet.Offer does in TestMode without a selector).
    private readonly object _rewardGate = new();
    private System.Threading.Tasks.TaskCompletionSource? _customRewardResolve;
    private System.Threading.Tasks.TaskCompletionSource _customRewardSignal = NewSignal();

    private static System.Threading.Tasks.TaskCompletionSource NewSignal() =>
        new(System.Threading.Tasks.TaskCreationOptions.RunContinuationsAsynchronously);

    private System.Threading.Tasks.Task CustomRewardSignal
    {
        get { lock (_rewardGate) { return _customRewardSignal.Task; } }
    }

    // The Crystal Sphere event minigame currently awaiting the agent's cell-clicks, or null. The
    // game drives this through a UI screen (NCrystalSphereScreen) that is null headless; a Harmony
    // patch on NCrystalSphereScreen.ShowScreen routes the minigame here instead (see
    // OnCrystalSphereScreenShown) and the offering event-option task suspends inside PlayMinigame on
    // its own completion source until the agent spends every divination. The minigame logic itself
    // (CrystalSphereMinigame) is plain C#; only the screen was UI.
    private readonly object _crystalGate = new();
    private MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? _pendingCrystalSphere;
    private System.Threading.Tasks.TaskCompletionSource _crystalSphereSignal = NewSignal();

    // Process-wide hook the ShowScreen Harmony patch calls; the latest run owns it (like testSelector).
    internal static System.Action<MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame>?
        CrystalSphereScreenHook;

    private System.Threading.Tasks.Task CrystalSphereSignal
    {
        get { lock (_crystalGate) { return _crystalSphereSignal.Task; } }
    }

    /// <summary>The Crystal Sphere minigame awaiting cell-clicks, or null. Read by the projection.</summary>
    internal MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? PendingCrystalSphere
    {
        get { lock (_crystalGate) { return _pendingCrystalSphere; } }
    }

    /// <summary>True while a Crystal Sphere minigame is awaiting the agent's divinations.</summary>
    private bool CrystalSpherePending
    {
        get { lock (_crystalGate) { return _pendingCrystalSphere is not null; } }
    }

    /// <summary>True while a custom (blocking) reward set is awaiting the agent's resolution.</summary>
    private bool CustomRewardPending
    {
        get { lock (_rewardGate) { return _customRewardResolve is not null; } }
    }

    /// <summary>
    /// True when an effect is suspended waiting on the agent — either a mid-effect card choice or
    /// a custom rewards set. The combat/event pumps return control (rather than draining) so the
    /// suspended decision surfaces instead of deadlocking.
    /// </summary>
    private bool EffectSuspended => Selector.Pending is not null || CustomRewardPending || CrystalSpherePending;

    /// <summary>
    /// Create and start a fresh single-player run with the given seed.
    /// Mirrors the game's own NSceneBootstrapper flow, minus all UI/asset steps.
    /// </summary>
    public static GameHost StartNewRun(string seed, int ascension = 0) =>
        StartNewRun(seed, playerCount: 1, ascension);

    /// <summary>
    /// Create and start a fresh run with <paramref name="playerCount"/> local players ("fake
    /// multiplayer"): the game's singleplayer net service hosting N players, all driven in-process by
    /// the harness. The game itself supports this configuration (see
    /// <c>RunManager.IsSingleplayerOrFakeMultiplayer</c> — turn/choice waits don't block on remote
    /// peers because every player acts locally). Players are assigned NetIds 1..N and successive
    /// characters from <c>ModelDb.AllCharacters</c> (wrapping if more players than characters).
    /// </summary>
    public static GameHost StartNewRun(string seed, int playerCount, int ascension = 0)
    {
        if (playerCount < 1)
        {
            throw new System.ArgumentOutOfRangeException(
                nameof(playerCount), playerCount, "A run needs at least one player.");
        }

        // Default characters: successive entries of ModelDb.AllCharacters (wrapping if more
        // players than characters), preserving the original assignment.
        List<CharacterModel> all = ModelDb.AllCharacters.ToList();
        var characters = new List<CharacterModel>(playerCount);
        for (int i = 0; i < playerCount; i++)
        {
            characters.Add(all[i % all.Count]);
        }
        return StartNewRun(seed, characters, ascension);
    }

    /// <summary>
    /// Create and start a fresh run with an explicit character per player. The first character is
    /// the local player (NetId 1); each subsequent one is another local ("fake multiplayer") player.
    /// Use this to pick the character(s) for the run (e.g. a front-end's character-select screen);
    /// the <c>playerCount</c> overloads assign characters automatically. Pick the models from
    /// <c>ModelDb.AllCharacters</c> (call <see cref="GameRuntime.EnsureInitialized"/> first so the
    /// model database is populated).
    /// </summary>
    public static GameHost StartNewRun(string seed, IReadOnlyList<CharacterModel> characters, int ascension = 0)
    {
        if (characters is null || characters.Count < 1)
        {
            throw new System.ArgumentException("A run needs at least one character.", nameof(characters));
        }

        GameRuntime.EnsureInitialized();

        // The game keeps run/combat state in process-wide singletons, so only one run
        // can exist per process. Tear down any previous run before starting a new one.
        if (RunManager.Instance.IsInProgress)
        {
            RunManager.Instance.CleanUp();
        }

        // The harness plays the full game: treat every epoch as unlocked so all content
        // (cards/relics/events) is available and the run starts with the Neow ancient event
        // (StartedWithNeow is derived from the run's unlock state — see RunManager).
        UnlockState unlock = UnlockState.all;

        var players = new List<Player>(characters.Count);
        for (int i = 0; i < characters.Count; i++)
        {
            players.Add(Player.CreateForNewRun(characters[i], unlock, (ulong)(i + 1)));
        }

        List<ActModel> acts = ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList();
        RunState runState = RunState.CreateForNewRun(
            players,
            acts,
            new List<ModifierModel>(),
            GameMode.Standard,
            ascension,
            seed);

        RunManager.Instance.SetUpNewSingleplayer(runState, shouldSave: false);
        RunManager.Instance.Launch();

        return InstallHarnessAndCreateHost(runState, seed);
    }

    /// <summary>
    /// Install the harness's process-wide seams (card selector, custom-reward gate, Crystal Sphere
    /// screen hook) for the given run and return the <see cref="GameHost"/> that drives it. Shared by
    /// <see cref="StartNewRun(string, int, int)"/> and <see cref="Restore"/>.
    /// </summary>
    private static GameHost InstallHarnessAndCreateHost(RunState runState, string seed)
    {
        // Install our card selector so mid-effect choices route to the harness. Disposing the
        // previous scope clears the process-wide selector stack before we claim it.
        _selectorScope?.Dispose();
        var selector = new HarnessCardSelector();
        _selectorScope = MegaCrit.Sts2.Core.Commands.CardSelectCmd.UseSelector(selector);

        var host = new GameHost { Run = runState, Seed = seed, Selector = selector };

        // Intercept relic/event custom reward sets (RewardsSet.Offer): without a selector the
        // game auto-takes them all in TestMode. Route them through the harness so the agent
        // chooses. The hook is process-wide; the latest run owns it.
        MegaCrit.Sts2.Core.Rewards.RewardsSet.testSelector = host.OnCustomRewardsOffered;

        // Intercept the Crystal Sphere event minigame's UI screen (null headless) so its grid
        // surfaces as agent choices instead. Process-wide; the latest run owns it.
        CrystalSphereScreenHook = host.OnCrystalSphereScreenShown;

        return host;
    }

    // ---------------------------------------------------------------------------------
    // Snapshot & restore (M7). A run serializes to the game's own SerializableRun save model
    // (RunManager.ToSave); restoring rebuilds the RunState (RunState.FromSerializable) and re-enters
    // the run the way the game's load path does (SetUpSavedSingleplayer → Launch → GenerateMap →
    // LoadIntoLatestMapCoord), minus all UI/asset loading. A snapshot taken on the map passes the
    // just-finished current room as the save's "pre-finished room" so restore re-enters it as already
    // completed (rather than re-running the combat). Snapshotting mid-combat is not supported yet
    // (combat state lives in CombatManager, not RunState).
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Capture a serializable snapshot of the run (the game's own <c>SerializableRun</c> save model).
    /// Intended for use out of combat (on the map / between rooms); the current room is recorded as
    /// the save's pre-finished room so <see cref="Restore"/> resumes without re-running it.
    /// </summary>
    public MegaCrit.Sts2.Core.Saves.SerializableRun Snapshot()
    {
        if (InCombat)
        {
            throw new InvalidOperationException(
                "Snapshotting mid-combat is not supported — snapshot out of combat (on the map).");
        }
        return RunManager.Instance.ToSave(Run.CurrentRoom);
    }

    /// <summary>
    /// Restore a run from a <see cref="Snapshot"/>, replacing any run in progress. Rebuilds the
    /// <c>RunState</c> and re-enters it through the logic half of the game's load path. Returns a
    /// fresh <see cref="GameHost"/> driving the restored run.
    /// </summary>
    public static GameHost Restore(MegaCrit.Sts2.Core.Saves.SerializableRun save, string seed)
    {
        GameRuntime.EnsureInitialized();
        if (RunManager.Instance.IsInProgress)
        {
            RunManager.Instance.CleanUp();
        }

        RunState runState = RunState.FromSerializable(save);
        Pump(RunManager.Instance.SetUpSavedSingleplayer(runState, save));
        RunManager.Instance.Launch();
        Pump(RunManager.Instance.GenerateMap());
        MegaCrit.Sts2.Core.Rooms.AbstractRoom? preFinished =
            MegaCrit.Sts2.Core.Rooms.AbstractRoom.FromSerializable(save.PreFinishedRoom, runState);
        Pump(RunManager.Instance.LoadIntoLatestMapCoord(preFinished));

        GameHost host = InstallHarnessAndCreateHost(runState, seed);
        host.WaitForEventReady();
        return host;
    }

    /// <summary>
    /// Serialize a <see cref="Snapshot"/> to the game's own save JSON (the format the game persists
    /// runs in). Out of combat only (see <see cref="Snapshot"/>); the seed is not stored in the run
    /// save, so keep it alongside if you need it to <see cref="RestoreFromJson"/>.
    /// </summary>
    public string ToSaveJson() =>
        MegaCrit.Sts2.Core.Saves.JsonSerializationUtility.ToJson(Snapshot());

    /// <summary>
    /// Restore a run from save JSON produced by <see cref="ToSaveJson"/> (or the game's own run save),
    /// replacing any run in progress. Returns a fresh <see cref="GameHost"/> driving the restored run.
    /// </summary>
    public static GameHost RestoreFromJson(string json, string seed)
    {
        MegaCrit.Sts2.Core.Saves.ReadSaveResult<MegaCrit.Sts2.Core.Saves.SerializableRun> result =
            MegaCrit.Sts2.Core.Saves.JsonSerializationUtility.FromJson<MegaCrit.Sts2.Core.Saves.SerializableRun>(json);
        if (!result.Success || result.SaveData is null)
        {
            throw new InvalidOperationException(
                $"Could not read save JSON: status={result.Status} {result.ErrorMessage}");
        }
        return Restore(result.SaveData, seed);
    }

    /// <summary>
    /// The seam invoked (via a Harmony patch on <c>NCrystalSphereScreen.ShowScreen</c>) when the
    /// Crystal Sphere event would open its UI minigame. Records the live <c>CrystalSphereMinigame</c>
    /// so it surfaces as <see cref="GamePhase.CrystalSphere"/>; the offering event-option task then
    /// suspends inside <c>PlayMinigame</c> on the minigame's completion source until the agent spends
    /// every divination (see <see cref="ClickCrystalSphereCell"/>). Runs on that option task's thread.
    /// </summary>
    internal void OnCrystalSphereScreenShown(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame grid)
    {
        lock (_crystalGate)
        {
            if (_pendingCrystalSphere is not null)
            {
                throw new InvalidOperationException("A Crystal Sphere minigame is already pending.");
            }
            _pendingCrystalSphere = grid;
            _crystalSphereSignal.TrySetResult();
        }
    }

    /// <summary>
    /// The game's <c>RewardsSet.testSelector</c> seam: invoked by <c>RewardsSet.Offer</c> (used by
    /// relic/event custom rewards) on the offering effect's thread-pool task. Surfaces the set as
    /// <see cref="GamePhase.Reward"/> and blocks that task until the agent takes/skips and proceeds,
    /// at which point <see cref="ProceedFromRewards"/> completes the returned task so the effect
    /// resumes. Mirrors the card-choice injection (<see cref="HarnessCardSelector"/>).
    /// </summary>
    internal System.Threading.Tasks.Task OnCustomRewardsOffered(MegaCrit.Sts2.Core.Rewards.RewardsSet set)
    {
        System.Threading.Tasks.TaskCompletionSource resolve = NewSignal();
        lock (_rewardGate)
        {
            if (PendingRewards is not null)
            {
                throw new InvalidOperationException(
                    "A rewards set is already pending when a custom reward was offered.");
            }
            PendingRewards = set;
            _customRewardResolve = resolve;
            _customRewardSignal.TrySetResult();
        }
        return resolve.Task;
    }

    /// <summary>
    /// Advance from a freshly-launched run into the first act's opening room
    /// (mirrors NSceneBootstrapper's Unassigned/EnterAct path). Async game work is
    /// pumped to completion synchronously.
    /// </summary>
    public void EnterFirstRoom()
    {
        RunManager rm = RunManager.Instance;
        Pump(rm.SetActInternal(0));
        rm.RunLocationTargetedBuffer.OnLocationChanged(Run.RunLocation);
        rm.MapSelectionSynchronizer.OnLocationChanged(Run.MapLocation);
        Pump(rm.EnterAct(0));
        // With all epochs unlocked the run opens on the Neow ancient event; wait for its
        // options to be generated (BeginEvent runs as a fire-and-forget task).
        WaitForEventReady();
    }

    /// <summary>The current combat, or null if not in a battle.</summary>
    public MegaCrit.Sts2.Core.Combat.CombatState? Combat =>
        RunManager.Instance.IsInProgress
            ? MegaCrit.Sts2.Core.Combat.CombatManager.Instance.DebugOnlyGetState()
            : null;

    /// <summary>True while a battle is in progress.</summary>
    public bool InCombat => MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress;

    /// <summary>
    /// Move to a reachable map coordinate, entering its room. In headless mode this
    /// uses the logic path (RunManager.EnterMapCoord), not the map-screen UI.
    /// </summary>
    public void MoveTo(MegaCrit.Sts2.Core.Map.MapCoord coord)
    {
        Pump(RunManager.Instance.EnterMapCoord(coord));
        // If we entered an event room, wait for its options to be generated before returning.
        WaitForEventReady();
        // If we entered a treasure room, open the chest (grant gold + offer any extra rewards) so
        // its relics surface as a choice.
        TryOpenTreasureChest();
    }

    /// <summary>
    /// Move on the map as the given player. In a single-player run this is the direct
    /// <see cref="MoveTo(MegaCrit.Sts2.Core.Map.MapCoord)"/>. In a multi-player run the party moves
    /// together by vote: this registers the player's vote (<c>MapSelectionSynchronizer.
    /// PlayerVotedForMapCoord</c>), and only once *every* player has voted does the game pick a
    /// destination (randomly weighted among the votes if they differ) and move — driven through the
    /// faithful <c>MoveToMapCoordAction</c> (which, in TestMode, calls the same <c>EnterMapCoord</c>).
    /// So applying a non-final player's move just records their vote and returns; the last vote moves.
    /// </summary>
    public void MoveTo(Player player, MegaCrit.Sts2.Core.Map.MapCoord coord)
    {
        if (Run.Players.Count <= 1)
        {
            MoveTo(coord);
            return;
        }

        MegaCrit.Sts2.Core.Multiplayer.Game.MapSelectionSynchronizer sync =
            RunManager.Instance.MapSelectionSynchronizer;
        var vote = new MegaCrit.Sts2.Core.Multiplayer.Game.MapVote
        {
            coord = coord,
            mapGenerationCount = sync.MapGenerationCount,
        };
        sync.PlayerVotedForMapCoord(player, Run.MapLocation, vote);
        // If that was the last vote, the synchronizer enqueued the move; run it. (A no-op for a
        // non-final voter, whose vote is just recorded.) Room-entry follow-ups run after the move.
        DrainActionQueue();
        WaitForEventReady();
        TryOpenTreasureChest();
    }

    /// <summary>
    /// Test/dev seam: enter a combat room for an arbitrary encounter directly, bypassing map
    /// navigation, so every act fight can be exercised in isolation. Mirrors the logic the game's
    /// own <c>RunManager.EnterMapPointInternal</c> runs for a combat point (pause the action
    /// executor, then enter the room via the sanctioned <c>EnterRoomDebug</c> test path). The
    /// encounter is cloned to a fresh mutable instance; returns once combat is set up and ready.
    /// </summary>
    public MegaCrit.Sts2.Core.Rooms.CombatRoom EnterEncounterDebug(
        MegaCrit.Sts2.Core.Models.EncounterModel encounter)
    {
        MegaCrit.Sts2.Core.Models.EncounterModel mutable =
            encounter.IsMutable ? encounter : encounter.ToMutable();
        RunManager.Instance.ActionExecutor.Pause();
        var room = (MegaCrit.Sts2.Core.Rooms.CombatRoom)Pump(
            RunManager.Instance.EnterRoomDebug(
                mutable.RoomType, MegaCrit.Sts2.Core.Map.MapPointType.Unassigned, mutable, showTransition: false));
        DrainActionQueue();
        return room;
    }

    /// <summary>
    /// Test/dev seam: grant the local player a relic and pump its on-obtain effect to quiescence or
    /// a surfaced choice/reward — the same way the treasure path obtains relics. Unlike awaiting
    /// <c>RelicCmd.Obtain</c> inline (which deadlocks if the relic's <c>AfterObtained</c> raises a
    /// custom reward through the harness gate, e.g. Orrery/Cauldron/CallingBell), the obtain runs as
    /// a fire-and-forget task pumped until it finishes or suspends, so any on-obtain reward surfaces
    /// as <see cref="GamePhase.Reward"/> for the agent to resolve. Returns the obtained mutable relic.
    /// </summary>
    public MegaCrit.Sts2.Core.Models.RelicModel ObtainRelicDebug(
        MegaCrit.Sts2.Core.Models.RelicModel relic)
    {
        MegaCrit.Sts2.Core.Models.RelicModel mutable = relic.IsMutable ? relic : relic.ToMutable();
        System.Threading.Tasks.Task obtain = MegaCrit.Sts2.Core.Helpers.TaskHelper.RunSafely(
            MegaCrit.Sts2.Core.Commands.RelicCmd.Obtain(mutable, Run.Players[0]));
        // Keep the obtain task registered while it is still in flight — whether it suspended on a
        // custom reward (resumed by ProceedFromRewards) or on a mid-effect card choice (resumed by
        // Apply(SelectCards), e.g. NewLeaf's transform or PreservedFog's removal pick) — so the
        // resumer pumps it to completion instead of leaving the continuation racing GetState.
        _suspendedRoomTask = obtain;
        PumpRoomTaskUntilIdleOrChoice(obtain);
        if (obtain.IsCompleted)
        {
            _suspendedRoomTask = null;
        }
        return mutable;
    }

    /// <summary>
    /// Test/dev seam: enter an event room for an arbitrary event directly, bypassing map
    /// navigation, so every act event can be exercised in isolation. Mirrors the harness's normal
    /// room-entry handling (wait for the event's options to be generated). The event is cloned to a
    /// fresh mutable instance.
    /// </summary>
    public MegaCrit.Sts2.Core.Rooms.EventRoom EnterEventDebug(
        MegaCrit.Sts2.Core.Models.EventModel ev)
    {
        // Unlike a CombatRoom (which needs a mutable encounter), an EventRoom takes the *canonical*
        // event and makes its own per-player mutable copy, so pass the canonical model through.
        MegaCrit.Sts2.Core.Models.EventModel canonical =
            ev.IsCanonical ? ev : MegaCrit.Sts2.Core.Models.ModelDb.GetById<MegaCrit.Sts2.Core.Models.EventModel>(ev.Id);
        RunManager.Instance.ActionExecutor.Pause();
        var room = (MegaCrit.Sts2.Core.Rooms.EventRoom)Pump(
            RunManager.Instance.EnterRoomDebug(
                MegaCrit.Sts2.Core.Rooms.RoomType.Event, MegaCrit.Sts2.Core.Map.MapPointType.Unassigned, canonical, showTransition: false));
        WaitForEventReady();
        return room;
    }

    /// <summary>True while the current (out-of-combat) room is an event room.</summary>
    private bool InEventRoom =>
        RunManager.Instance.IsInProgress
        && !InCombat
        && Run.CurrentRoom is MegaCrit.Sts2.Core.Rooms.EventRoom;

    /// <summary>
    /// The local player's mutable event when the current room is an (out-of-combat) event room,
    /// or null otherwise. Used to surface <see cref="GamePhase.Event"/> options in the projection.
    /// </summary>
    internal MegaCrit.Sts2.Core.Models.EventModel? CurrentEvent =>
        InEventRoom ? RunManager.Instance.EventSynchronizer.GetLocalEvent() : null;

    /// <summary>
    /// The given player's own mutable event when the current room is an event room, or null
    /// otherwise. In a multi-player run every player gets their own instance of the room's event and
    /// resolves it independently (<c>EventSynchronizer.GetEventForPlayer</c>).
    /// </summary>
    internal MegaCrit.Sts2.Core.Models.EventModel? CurrentEventForPlayer(Player player) =>
        InEventRoom ? RunManager.Instance.EventSynchronizer.GetEventForPlayer(player) : null;

    private static bool EventIsActionable(MegaCrit.Sts2.Core.Models.EventModel? e) =>
        e is not null && e.CurrentOptions.Any(o => !o.IsLocked && !o.IsProceed);

    /// <summary>
    /// True when *any* player's event is awaiting a real (non-proceed, unlocked) choice — so the run
    /// is still in an actionable event room. A finished event — or one down to only a "proceed"
    /// option — is not actionable: the party leaves by moving on the map.
    /// </summary>
    internal bool HasActionableEvent =>
        InEventRoom && Run.Players.Any(HasActionableEventForPlayer);

    /// <summary>
    /// True when the given player still has a choice to make in the current event. For a per-player
    /// event that means the event has a real (non-proceed, unlocked) option. For a shared (vote-based)
    /// event the same, but a player who has already cast their vote this round is waiting for the
    /// others — not actionable — until every vote is in and the option resolves.
    /// </summary>
    internal bool HasActionableEventForPlayer(Player player)
    {
        MegaCrit.Sts2.Core.Models.EventModel? e = CurrentEventForPlayer(player);
        if (!EventIsActionable(e))
        {
            return false;
        }
        if (e!.IsShared && RunManager.Instance.EventSynchronizer.GetPlayerVote(player).HasValue)
        {
            return false; // already voted this round; awaiting the other players
        }
        return true;
    }

    /// <summary>The current room as a treasure room, or null when not standing in one.</summary>
    internal MegaCrit.Sts2.Core.Rooms.TreasureRoom? CurrentTreasureRoom =>
        RunManager.Instance.IsInProgress && !InCombat
            ? Run.CurrentRoom as MegaCrit.Sts2.Core.Rooms.TreasureRoom
            : null;

    private MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer TreasureSync =>
        RunManager.Instance.TreasureRoomRelicSynchronizer;

    /// <summary>
    /// True while standing in a treasure room whose chest is open and which still has relics to
    /// pick. Surfaces as <see cref="GamePhase.Treasure"/> with take/skip options. A skipped or
    /// emptied chest leaves no relics, so the player is back on the map.
    /// </summary>
    internal bool HasTreasureChoice =>
        CurrentTreasureRoom is not null
        && PendingRewards is null
        && Selector.Pending is null
        && TreasureSync.CurrentRelics is { Count: > 0 };

    /// <summary>
    /// True when the given player still has a treasure pick to make. Single-player: the same as
    /// <see cref="HasTreasureChoice"/>. Multi-player: the chest is a vote — each player picks one relic
    /// (or skips), so a player who has already cast their pick is waiting for the others (not
    /// actionable) until everyone has, when the game awards the relics (conflicts resolved by it).
    /// </summary>
    internal bool HasTreasureChoiceForPlayer(Player player)
    {
        if (!HasTreasureChoice)
        {
            return false;
        }
        if (Run.Players.Count <= 1)
        {
            return true;
        }
        return !TreasureSync.GetPlayerVote(player).voteReceived;
    }

    /// <summary>The current room as a rest site, or null when not standing in one.</summary>
    internal MegaCrit.Sts2.Core.Rooms.RestSiteRoom? CurrentRestSiteRoom =>
        RunManager.Instance.IsInProgress && !InCombat
            ? Run.CurrentRoom as MegaCrit.Sts2.Core.Rooms.RestSiteRoom
            : null;

    private MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer RestSiteSync =>
        RunManager.Instance.RestSiteSynchronizer;

    /// <summary>
    /// True while at a rest site that still has at least one usable action. Choosing one consumes
    /// the rest action (the game clears the remaining options), after which the player is back on
    /// the map.
    /// </summary>
    internal bool HasRestChoice =>
        CurrentRestSiteRoom is not null
        && PendingRewards is null
        && Selector.Pending is null
        && Run.Players.Any(p => RestSiteSync.GetOptionsForPlayer(p.NetId).Any(o => o.IsEnabled));

    /// <summary>
    /// True while the given player still has a usable rest action. Each player rests independently
    /// (their own rest/smith/…); choosing one consumes that player's rest (the game clears their
    /// remaining options). The party leaves the rest site by moving on the map once every player is done.
    /// </summary>
    internal bool HasRestChoiceForPlayer(Player player) =>
        CurrentRestSiteRoom is not null
        && PendingRewards is null
        && Selector.Pending is null
        && RestSiteSync.GetOptionsForPlayer(player.NetId).Any(o => o.IsEnabled);

    /// <summary>The current room as a merchant shop, or null when not standing in one.</summary>
    internal MegaCrit.Sts2.Core.Rooms.MerchantRoom? CurrentMerchantRoom =>
        RunManager.Instance.IsInProgress && !InCombat
            ? Run.CurrentRoom as MegaCrit.Sts2.Core.Rooms.MerchantRoom
            : null;

    /// <summary>
    /// True while standing in a merchant shop and able to act on it. A shop is never "consumed":
    /// the player can keep buying affordable items or leave by moving on the map, so this stays
    /// true for the whole visit (unless a reward/card choice is mid-resolution).
    /// </summary>
    internal bool HasShopChoice =>
        CurrentMerchantRoom is not null
        && PendingRewards is null
        && Selector.Pending is null;

    /// <summary>
    /// Classify a merchant entry into (item-type, model-id, card) for the option/projection layer.
    /// The card is non-null only for card entries.
    /// </summary>
    internal static (string type, string id, MegaCrit.Sts2.Core.Models.CardModel? card) ClassifyShopEntry(
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry) => entry switch
    {
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardEntry c =>
            ("Card", c.CreationResult?.Card.Id.Entry ?? "?", c.CreationResult?.Card),
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantRelicEntry r =>
            ("Relic", r.Model?.Id.Entry ?? "?", null),
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantPotionEntry p =>
            ("Potion", p.Model?.Id.Entry ?? "?", null),
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardRemovalEntry =>
            ("CardRemoval", "CardRemoval", null),
        _ => ("Unknown", "?", null),
    };

    /// <summary>
    /// Play a card from hand at an optional target via the canonical player path
    /// (CardModel.TryManualPlay), which validates targeting, spends energy, and runs
    /// the card's effects, then drains the action queue. Returns false if the play was
    /// rejected (e.g. not enough energy / invalid target). Pass the target for
    /// AnyEnemy/AnyAlly cards; null otherwise.
    /// </summary>
    public bool PlayCard(MegaCrit.Sts2.Core.Models.CardModel card, MegaCrit.Sts2.Core.Entities.Creatures.Creature? target)
    {
        bool enqueued = card.TryManualPlay(target);
        if (!enqueued)
        {
            return false;
        }
        // The play runs on the thread pool; pump until it finishes or blocks on a card choice.
        PumpCombatUntilIdleOrChoice();
        TryOfferCombatRewards();
        return true;
    }

    /// <summary>
    /// End the turn and resolve the enemy turn to quiescence. In the single-process
    /// "fake-multiplayer" model the game treats the turn as endable as soon as one player ends
    /// (<c>CombatManager.AllPlayersReadyToEndTurn</c> is unconditionally true when
    /// <c>IsSingleplayerOrFakeMultiplayer</c>), so ending *any* player ends the shared round and
    /// starts the enemy turn — the other players act during the same Play phase *before* this end,
    /// they do not take independent turns. The enemy turn runs on background tasks (not the player
    /// action queue), so we wait via the combat events until the players can act again or combat ends.
    /// </summary>
    public void EndTurn(Player player)
    {
        MegaCrit.Sts2.Core.Commands.PlayerCmd.EndTurn(player, canBackOut: false);
        DrainActionQueue();
        if (MegaCrit.Sts2.Core.Combat.CombatManager.Instance.AllPlayersReadyToEndTurn())
        {
            WaitUntilPlayerCanActOrCombatEnds(player);
            TryOfferCombatRewards();
        }
    }

    // ---------------------------------------------------------------------------------
    // Potions. Using a potion enqueues a UsePotionAction via the faithful manual-use path
    // (PotionModel.EnqueueManualUse, the same the UI's potion popup drives); discarding enqueues a
    // DiscardPotionGameAction. Both run on the action queue in or out of combat, so we pump it to
    // quiescence (a potion that raises a card choice — e.g. a discovery potion — surfaces it, and one
    // that ends combat triggers its rewards).
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Use (drink/throw) the potion the option points at, optionally at its target, then pump the
    /// action queue to quiescence (or to a raised card choice). Offers combat rewards if the potion
    /// ended the fight.
    /// </summary>
    private void UsePotion(GameOption option)
    {
        MegaCrit.Sts2.Core.Models.PotionModel potion = option.PotionModel
            ?? throw new InvalidOperationException("Use-potion option carried no potion.");
        potion.EnqueueManualUse(option.Target);
        PumpCombatUntilIdleOrChoice();
        TryOfferCombatRewards();
    }

    /// <summary>
    /// Discard the potion in the option's belt slot (no effect), then drain the action queue. Works
    /// in and out of combat (the action records which, for turn-order correctness).
    /// </summary>
    private void DiscardPotion(GameOption option)
    {
        int slot = option.PotionSlot
            ?? throw new InvalidOperationException("Discard-potion option carried no slot.");
        var action = new MegaCrit.Sts2.Core.GameActions.DiscardPotionGameAction(
            option.Player!, (uint)slot, InCombat);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(action);
        PumpCombatUntilIdleOrChoice();
    }

    // ---------------------------------------------------------------------------------
    // Post-combat rewards. The faithful victory→rewards flow is driven by the combat UI
    // node (NCombatUi.OnCombatWon → CombatRoom.OfferRoomEndRewards), which is null in the
    // headless harness. We reproduce its logic half: once a won combat has fully ended we
    // generate the room's RewardsSet and register it with the synchronizer, then surface it
    // as GamePhase.Reward. Selecting/skipping happens through the public option API.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// If combat has just been won, generate and offer the room's end-of-combat rewards so
    /// they surface as <see cref="GamePhase.Reward"/>. Idempotent per combat room; a no-op
    /// while still in combat, on a loss, or for encounters that grant no rewards.
    /// </summary>
    private void TryOfferCombatRewards()
    {
        if (InCombat || PendingRewards is not null)
        {
            return;
        }
        if (Run.CurrentRoom is not MegaCrit.Sts2.Core.Rooms.CombatRoom room || ReferenceEquals(room, _rewardedRoom))
        {
            return;
        }
        // The team won unless every player is dead (all dead = a loss, handled as game over).
        if (Run.Players.All(p => !p.Creature.IsAlive))
        {
            return;
        }
        if (room.Encounter is { ShouldGiveRewards: false })
        {
            return;
        }

        _rewardedRoom = room;

        // The game gives each (alive) player their own end-of-combat rewards (CombatRoom.
        // OfferRoomEndRewards loops the players). GenerateForRoomEnd populates a set + runs its
        // reward-modifying hooks without offering it; BeginRewardsSet registers it for the
        // select/skip APIs. The first player's set is surfaced now; the rest are queued and surface
        // one at a time as each player proceeds (see ProceedFromRewards).
        var sets = new List<MegaCrit.Sts2.Core.Rewards.RewardsSet>();
        foreach (Player player in Run.Players)
        {
            if (!player.Creature.IsAlive)
            {
                continue;
            }
            MegaCrit.Sts2.Core.Rewards.RewardsSet set =
                Pump(MegaCrit.Sts2.Core.Commands.RewardsCmd.GenerateForRoomEnd(player, room));
            _ = RunManager.Instance.RewardsSetSynchronizer.BeginRewardsSet(set);
            sets.Add(set);
        }

        PendingRewards = sets[0];
        for (int i = 1; i < sets.Count; i++)
        {
            _queuedRewardSets.Enqueue(sets[i]);
        }
    }

    /// <summary>True when the given player is the local player (NetId 1).</summary>
    private bool IsLocalPlayer(Player player) => player.NetId == Run.Players[0].NetId;

    private static T Pump<T>(System.Threading.Tasks.Task<T> task) => task.GetAwaiter().GetResult();

    // ---------------------------------------------------------------------------------
    // Public API: read state, list legal options, apply a chosen option.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Capture an immutable snapshot of the current mechanical game state. The snapshot
    /// holds no live references, so it remains valid after the game advances.
    /// </summary>
    public GameState GetState() => GameStateProjection.Capture(this);

    /// <summary>
    /// Enumerate the legal actions available to the given player in the current state.
    /// Currently covers combat (playable cards × legal targets, end turn) and map moves;
    /// other room/screen choices arrive with later milestones. Each option is valid only
    /// for the state it was listed from — apply it before advancing the game.
    /// </summary>
    public IReadOnlyList<GameOption> ListOptions(ulong playerId)
    {
        Player player = GetPlayer(playerId);
        var options = new List<GameOption>();

        // A mid-effect card choice takes precedence over everything else: until it is
        // resolved, the game is blocked and no other action can be taken.
        PendingChoice? pending = Selector.Pending;
        if (pending is not null)
        {
            return BuildChoiceOptions(player, pending);
        }

        // The Crystal Sphere event minigame: spend divinations clearing cells (then the revealed
        // items' rewards surface through the normal custom-reward screen).
        if (PendingCrystalSphere is { } minigame)
        {
            return BuildCrystalSphereOptions(player, minigame);
        }

        // The post-combat rewards screen: take rewards then proceed back to the map. The options are
        // attributed to the set's owner (in multi-player the players' rewards surface one at a time).
        if (PendingRewards is not null)
        {
            var rewardOptions = new List<GameOption>(BuildRewardOptions(PendingRewards));
            // The potion belt stays usable on the reward screen (out of combat, so only AnyTime
            // potions apply — CombatOnly ones are gated by CanUsePotionNow).
            AddPotionOptions(player, rewardOptions);
            return rewardOptions;
        }

        // A treasure room with its chest open: take one of the offered relics or skip. Multi-player
        // is a per-player vote — a player who has already picked waits for the others (no options).
        if (HasTreasureChoiceForPlayer(player))
        {
            return BuildTreasureOptions(player);
        }
        if (HasTreasureChoice)
        {
            return options; // this player has picked; awaiting the others
        }

        // A rest site: choose a rest action (rest/smith/…). Each player rests independently; one who
        // has already rested waits (no options) until everyone is done and the party moves on.
        if (HasRestChoiceForPlayer(player))
        {
            return BuildRestOptions(player);
        }
        if (HasRestChoice)
        {
            return options; // this player is done resting; awaiting the others
        }

        // A merchant shop: buy affordable items / card removal, or leave by moving on the map.
        if (HasShopChoice)
        {
            return BuildShopOptions(player);
        }

        // An event room awaiting a choice (e.g. the opening Neow ancient event). Each player
        // resolves their *own* event, so list this player's options.
        if (HasActionableEventForPlayer(player))
        {
            return BuildEventOptions(player, CurrentEventForPlayer(player)!);
        }

        // In a multi-player event room, a player who has already finished their own event waits for
        // the others before the party can move on (the map vote needs every player). Offer nothing
        // meanwhile, rather than letting this player vote to leave the room prematurely.
        if (HasActionableEvent)
        {
            return options; // empty: this player is done, others are still choosing
        }

        if (InCombat)
        {
            MegaCrit.Sts2.Core.Entities.Players.PlayerCombatState? pcs = player.PlayerCombatState;
            if (pcs is null || pcs.Phase != MegaCrit.Sts2.Core.Combat.PlayerTurnPhase.Play)
            {
                return options; // not this player's turn to act
            }

            MegaCrit.Sts2.Core.Combat.CombatState combat = Combat!;
            foreach (MegaCrit.Sts2.Core.Models.CardModel card in pcs.Hand.Cards)
            {
                if (!card.CanPlay())
                {
                    continue;
                }
                if (card.TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                {
                    foreach (MegaCrit.Sts2.Core.Entities.Creatures.Creature enemy in combat.HittableEnemies)
                    {
                        if (card.IsValidTarget(enemy))
                        {
                            // Project per target so the option shows the actual damage to that enemy
                            // (its Vulnerable/Intangible/etc. folded in).
                            CardView targetedView = GameStateProjection.ProjectCard(card, canPlay: true, target: enemy);
                            options.Add(GameOption.PlayCardOption(player, card, targetedView, enemy));
                        }
                    }
                }
                else
                {
                    CardView view = GameStateProjection.ProjectCard(card, canPlay: true);
                    options.Add(GameOption.PlayCardOption(player, card, view, target: null));
                }
            }

            AddPotionOptions(player, options);
            options.Add(GameOption.EndTurnOption(player));
            return options;
        }

        // Out of combat: map room choices, plus any anytime potion actions.
        foreach (MegaCrit.Sts2.Core.Map.MapPoint point in GameStateProjection.ReachablePoints(Run))
        {
            options.Add(GameOption.MoveToOption(player, point.coord));
        }
        AddPotionOptions(player, options);
        return options;
    }

    /// <summary>List options for the first (or only) player. Convenience for single-player.</summary>
    public IReadOnlyList<GameOption> ListOptions() => ListOptions(Run.Players[0].NetId);

    /// <summary>
    /// Append the given player's potion actions (use/discard) to <paramref name="options"/>. Usable
    /// in combat (during the Play phase) and out of combat on the map/shop; not offered on the
    /// reward/event/treasure/rest/choice screens (the game blocks potion use there). A targeted
    /// (AnyEnemy) potion in combat expands to one use option per valid enemy; everything else is a
    /// single untargeted use. Discard is offered whenever the potion can be removed.
    /// </summary>
    private void AddPotionOptions(Player player, List<GameOption> options)
    {
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Models.PotionModel?> slots = player.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
        {
            if (slots[i] is not { } potion)
            {
                continue;
            }
            if (CanUsePotionNow(potion))
            {
                if (InCombat && potion.TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                {
                    foreach (MegaCrit.Sts2.Core.Entities.Creatures.Creature enemy in Combat!.HittableEnemies)
                    {
                        if (potion.IsValidTarget(enemy))
                        {
                            options.Add(GameOption.UsePotionOption(player, i, potion, enemy));
                        }
                    }
                }
                else
                {
                    options.Add(GameOption.UsePotionOption(player, i, potion, target: null));
                }
            }
            if (CanDiscardPotionNow(potion))
            {
                options.Add(GameOption.DiscardPotionOption(player, i, potion));
            }
        }
    }

    /// <summary>
    /// True when the potion can be manually used right now: it is not already queued, its owner is
    /// alive and allowed to remove potions, it passes its custom usability check, and its usage
    /// window is open (AnyTime always; CombatOnly only in combat; None/Automatic never manually).
    /// </summary>
    private bool CanUsePotionNow(MegaCrit.Sts2.Core.Models.PotionModel potion)
    {
        if (potion.IsQueued || potion.Owner.Creature.IsDead
            || !potion.Owner.CanRemovePotions || !potion.PassesCustomUsabilityCheck)
        {
            return false;
        }
        return potion.Usage switch
        {
            MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.AnyTime => true,
            MegaCrit.Sts2.Core.Entities.Potions.PotionUsage.CombatOnly => InCombat,
            _ => false,
        };
    }

    /// <summary>True when the potion can be discarded now (not queued, owner alive and able to remove potions).</summary>
    private static bool CanDiscardPotionNow(MegaCrit.Sts2.Core.Models.PotionModel potion) =>
        !potion.IsQueued && !potion.Owner.Creature.IsDead && potion.Owner.CanRemovePotions;

    /// <summary>
    /// Build the options that resolve a pending card choice. Single-select choices enumerate
    /// one option per card (plus a skip when the choice allows selecting none). Multi-select
    /// choices (min &gt; 1) currently offer a single exact-minimum selection so play never
    /// blocks; full subset enumeration is future work.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildChoiceOptions(Player player, PendingChoice pending)
    {
        var options = new List<GameOption>();
        // A forge choice previews each card as its upgraded form.
        bool? upgradePreview = pending.IsUpgradeSelection ? true : (bool?)null;
        var views = pending.Options
            .Select(c => GameStateProjection.ProjectCard(c, canPlay: false, upgradePreview))
            .ToList();

        if (pending.MinSelect <= 1)
        {
            for (int i = 0; i < pending.Options.Count; i++)
            {
                options.Add(GameOption.SelectCardsOption(
                    player, new[] { pending.Options[i] }, new[] { views[i] }));
            }
            if (pending.MinSelect == 0)
            {
                options.Add(GameOption.SelectCardsOption(
                    player,
                    System.Array.Empty<MegaCrit.Sts2.Core.Models.CardModel>(),
                    System.Array.Empty<CardView>()));
            }
        }
        else
        {
            var cards = pending.Options.Take(pending.MinSelect).ToList();
            var cardViews = views.Take(pending.MinSelect).ToList();
            options.Add(GameOption.SelectCardsOption(player, cards, cardViews));
        }
        return options;
    }

    /// <summary>
    /// Build the options for the post-combat rewards screen: one take option per not-yet-taken
    /// reward (card rewards expand to one option per offered card), plus a proceed option that
    /// leaves the screen and skips whatever is left.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildRewardOptions(MegaCrit.Sts2.Core.Rewards.RewardsSet set)
    {
        Player player = set.Player;
        var options = new List<GameOption>();
        foreach (MegaCrit.Sts2.Core.Rewards.Reward reward in set.Rewards)
        {
            if (reward.SuccessfullySelected)
            {
                continue;
            }
            switch (reward)
            {
                case MegaCrit.Sts2.Core.Rewards.GoldReward gold:
                    options.Add(GameOption.TakeRewardOption(player, gold, $"Take {gold.Amount} gold"));
                    break;
                case MegaCrit.Sts2.Core.Rewards.PotionReward potion:
                    options.Add(GameOption.TakeRewardOption(
                        player, potion, $"Take potion {potion.Potion?.Id.Entry ?? "?"}"));
                    break;
                case MegaCrit.Sts2.Core.Rewards.RelicReward relic:
                    options.Add(GameOption.TakeRewardOption(
                        player, relic, $"Take relic {relic.Relic?.Id.Entry ?? "?"}"));
                    break;
                case MegaCrit.Sts2.Core.Rewards.CardReward card:
                    foreach (MegaCrit.Sts2.Core.Models.CardModel offered in card.Cards)
                    {
                        CardView view = GameStateProjection.ProjectCard(offered, canPlay: false);
                        options.Add(GameOption.TakeCardRewardOption(player, card, offered, view));
                    }
                    // The card reward's alternatives (Pael's Wing sacrifice, reroll, …). Plain "Skip"
                    // is omitted — ProceedFromRewards already skips untaken rewards — so only the ones
                    // with a real effect surface.
                    foreach (MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative alt
                             in MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative.Generate(card))
                    {
                        if (alt.OptionId == "Skip")
                        {
                            continue;
                        }
                        options.Add(GameOption.TakeCardRewardAlternativeOption(player, card, alt));
                    }
                    break;
            }
        }
        options.Add(GameOption.ProceedFromRewardsOption(player));
        return options;
    }

    /// <summary>
    /// Build the options for an event room: one per unlocked, non-proceed event option, in the
    /// event's own option order (the index is preserved so <see cref="Apply"/> can resolve it
    /// against the game's <c>EventSynchronizer.ChooseLocalOption</c> seam).
    /// </summary>
    private static IReadOnlyList<GameOption> BuildEventOptions(Player player, MegaCrit.Sts2.Core.Models.EventModel ev)
    {
        var options = new List<GameOption>();
        IReadOnlyList<MegaCrit.Sts2.Core.Events.EventOption> current = ev.CurrentOptions;
        for (int i = 0; i < current.Count; i++)
        {
            MegaCrit.Sts2.Core.Events.EventOption opt = current[i];
            if (opt.IsLocked || opt.IsProceed)
            {
                continue;
            }
            options.Add(GameOption.ChooseEventOption(player, i, opt));
        }
        return options;
    }

    /// <summary>
    /// Build the options for an opened treasure room: one take option per offered relic plus a
    /// skip option. Resolved against the synchronizer's relic indices.
    /// </summary>
    private IReadOnlyList<GameOption> BuildTreasureOptions(Player player)
    {
        var options = new List<GameOption>();
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Models.RelicModel>? relics =
            TreasureSync.CurrentRelics;
        if (relics is not null)
        {
            for (int i = 0; i < relics.Count; i++)
            {
                options.Add(GameOption.TakeTreasureRelicOption(player, i, relics[i].Id.Entry));
            }
        }
        options.Add(GameOption.SkipTreasureOption(player));
        return options;
    }

    /// <summary>
    /// Build the options for a rest site: one per usable (enabled) option, carrying its live index
    /// so <see cref="Apply"/> can resolve it via <c>RestSiteSynchronizer.ChooseLocalOption</c>.
    /// Disabled options (e.g. Smith with no upgradable cards) are omitted.
    /// </summary>
    private IReadOnlyList<GameOption> BuildRestOptions(Player player)
    {
        var options = new List<GameOption>();
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption> rest =
            RestSiteSync.GetOptionsForPlayer(player.NetId);
        for (int i = 0; i < rest.Count; i++)
        {
            if (rest[i].IsEnabled)
            {
                options.Add(GameOption.ChooseRestOption(player, i, rest[i].OptionId));
            }
        }
        return options;
    }

    /// <summary>
    /// Build the options for a merchant shop: one buy option per in-stock, affordable item (cards,
    /// relics, potions, and the card-removal service), plus the reachable map moves so the player
    /// can leave. Unaffordable/out-of-stock items are omitted from the actionable set (they are
    /// still visible in <see cref="ShopView"/>).
    /// </summary>
    /// <summary>The given player's own merchant inventory (each player shops their own stock/gold).</summary>
    private MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory InventoryFor(Player player) =>
        CurrentMerchantRoom!.Inventories[Run.GetPlayerSlotIndex(player)];

    private IReadOnlyList<GameOption> BuildShopOptions(Player player)
    {
        var options = new List<GameOption>();
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory inv = InventoryFor(player);
        foreach (MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry in inv.AllEntries)
        {
            if (!entry.IsStocked || !entry.EnoughGold)
            {
                continue;
            }
            // A potion can only be bought when the belt has a free slot; the game's purchase fails
            // (PotionProcureFailureReason.FailureSpace) otherwise, so don't offer it as buyable.
            if (entry is MegaCrit.Sts2.Core.Entities.Merchant.MerchantPotionEntry && !player.HasOpenPotionSlots)
            {
                continue;
            }
            (string type, string id, MegaCrit.Sts2.Core.Models.CardModel? card) = ClassifyShopEntry(entry);
            CardView? view = card is null ? null : GameStateProjection.ProjectCard(card, canPlay: false);
            options.Add(GameOption.BuyShopItemOption(player, entry, type, id, view));
        }

        // The shop is left by moving on the map (there is no separate proceed step headless).
        foreach (MegaCrit.Sts2.Core.Map.MapPoint point in GameStateProjection.ReachablePoints(Run))
        {
            options.Add(GameOption.MoveToOption(player, point.coord));
        }
        AddPotionOptions(player, options);
        return options;
    }

    /// <summary>
    /// Build the options for the Crystal Sphere minigame: one click option per still-hidden cell
    /// (cleared with the active tool), plus a switch-tool option for the tool not currently active.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildCrystalSphereOptions(
        Player player, MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame)
    {
        var options = new List<GameOption>();
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell[,] cells = minigame.cells;
        for (int x = 0; x < cells.GetLength(0); x++)
        {
            for (int y = 0; y < cells.GetLength(1); y++)
            {
                if (cells[x, y].IsHidden)
                {
                    options.Add(GameOption.ClickCrystalSphereCellOption(player, x, y));
                }
            }
        }

        // Offer the tool not currently selected so the agent can switch between area/single clears.
        var big = MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Big;
        var small = MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Small;
        options.Add(GameOption.SetCrystalSphereToolOption(
            player, minigame.CrystalSphereTool == big ? small : big));
        return options;
    }

    /// <summary>
    /// Resolve a chosen option against the live game and pump to quiescence. The option
    /// must have come from <see cref="ListOptions(ulong)"/> for the current state.
    /// </summary>
    public void Apply(GameOption option)
    {
        switch (option.Kind)
        {
            case OptionKind.PlayCard:
                if (!PlayCard(option.CardModel!, option.Target))
                {
                    throw new InvalidOperationException($"Card play was rejected: {option.Description}");
                }
                break;
            case OptionKind.EndTurn:
                EndTurn(option.Player!);
                break;
            case OptionKind.MoveTo:
                MoveTo(option.Player!, option.MapCoord!.Value);
                break;
            case OptionKind.SelectCards:
                if (Selector.Pending is null)
                {
                    throw new InvalidOperationException("No card choice is pending to resolve.");
                }
                // Whether this choice was raised during the enemy turn (PlayerTurnPhase.None) rather
                // than by the player's own card effect (PlayerTurnPhase.Play) — it determines how the
                // resumed effect settles below. Capture it before resolving (the resume is async).
                bool enemyTurnChoice = InCombat
                    && Run.Players[0].PlayerCombatState?.Phase != MegaCrit.Sts2.Core.Combat.PlayerTurnPhase.Play;
                Selector.Resolve(option.SelectedCardModels!);
                // The effect resumes on the thread pool; pump until it finishes or blocks again.
                // A choice raised mid-event or mid-rest-action resumes that effect's task rather
                // than the combat action queue, so pump the matching surface.
                if (InCombat)
                {
                    // An enemy-turn choice resumes the enemy turn (background tasks, not the player
                    // action queue), so wait for the turn to finish / the player to act / a further
                    // choice; a player-effect choice resumes through the action queue.
                    if (enemyTurnChoice)
                    {
                        WaitUntilPlayerCanActOrCombatEnds(Run.Players[0]);
                        TryOfferCombatRewards();
                    }
                    else
                    {
                        PumpCombatUntilIdleOrChoice();
                    }
                }
                else if (_restChoiceTask is { IsCompleted: false } restTask)
                {
                    PumpRoomTaskUntilIdleOrChoice(restTask);
                    if (_restChoiceTask.IsCompleted)
                    {
                        _restChoiceTask = null;
                    }
                }
                else if (_shopRemovalTask is { IsCompleted: false } shopTask)
                {
                    PumpRoomTaskUntilIdleOrChoice(shopTask);
                    FinishShopRemovalIfDone();
                }
                else if (_suspendedRoomTask is { IsCompleted: false } roomTask)
                {
                    // A fire-and-forget room/relic effect (e.g. a relic's on-obtain transform/removal
                    // pick from ObtainRelicDebug) raised this choice; resume it to completion.
                    PumpRoomTaskUntilIdleOrChoice(roomTask);
                    if (_suspendedRoomTask.IsCompleted)
                    {
                        _suspendedRoomTask = null;
                    }
                }
                else
                {
                    PumpEventUntilIdleOrChoice();
                }
                break;
            case OptionKind.ChooseEventOption:
                ApplyEventOption(option.Player!, option.EventOptionIndex!.Value);
                break;
            case OptionKind.TakeReward:
                TakeReward(option);
                break;
            case OptionKind.TakeCardRewardAlternative:
                TakeCardRewardAlternative(option);
                break;
            case OptionKind.ProceedFromRewards:
                ProceedFromRewards();
                break;
            case OptionKind.TakeTreasureRelic:
                PickTreasureRelicForPlayer(option.Player!, option.TreasureRelicIndex!.Value);
                break;
            case OptionKind.SkipTreasure:
                PickTreasureRelicForPlayer(option.Player!, null);
                break;
            case OptionKind.ChooseRestOption:
                ChooseRestOption(option.Player!, option.RestOptionIndex!.Value);
                break;
            case OptionKind.BuyShopItem:
                BuyShopItem(option);
                break;
            case OptionKind.UsePotion:
                UsePotion(option);
                break;
            case OptionKind.DiscardPotion:
                DiscardPotion(option);
                break;
            case OptionKind.ClickCrystalSphereCell:
                ClickCrystalSphereCell(option.CrystalSphereCell!.Value.Col, option.CrystalSphereCell.Value.Row);
                break;
            case OptionKind.SetCrystalSphereTool:
                SetCrystalSphereTool(option.CrystalSphereToolValue!.Value);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(option), option.Kind, "Unknown option kind.");
        }
    }

    // RewardsSetSynchronizer's select/skip have local-only public entries (SelectLocalReward /
    // SkipLocalRewardsSet); the per-player seams the net-message handler uses are private. In a
    // multi-player run a non-local player's reward set is driven through those directly.
    private static readonly System.Reflection.MethodInfo _selectRewardForPlayer =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer).GetMethod(
            "SelectRewardForPlayer",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic,
            binder: null, types: new[] { typeof(Player), typeof(int) }, modifiers: null)
        ?? throw new InvalidOperationException(
            "RewardsSetSynchronizer.SelectRewardForPlayer(Player, int) not found — the game's rewards API changed.");

    private static readonly System.Reflection.MethodInfo _skipRewardsForPlayer =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer).GetMethod(
            "SkipRewardsSetOnStackTopForPlayer",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic,
            binder: null, types: new[] { typeof(Player) }, modifiers: null)
        ?? throw new InvalidOperationException(
            "RewardsSetSynchronizer.SkipRewardsSetOnStackTopForPlayer(Player) not found — the game's rewards API changed.");

    /// <summary>
    /// Select <paramref name="reward"/> from the current pending set, routing by its owner: the local
    /// player uses the faithful <c>SelectLocalReward</c>; any other player goes through the per-player
    /// <c>SelectRewardForPlayer(player, index)</c> seam (the index into the set's reward list).
    /// </summary>
    private void SelectRewardForOwner(MegaCrit.Sts2.Core.Rewards.Reward reward)
    {
        MegaCrit.Sts2.Core.Rewards.RewardsSet set = PendingRewards!;
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        if (IsLocalPlayer(set.Player))
        {
            Pump(sync.SelectLocalReward(reward));
            return;
        }
        int index = IndexOfReward(set, reward);
        Pump((System.Threading.Tasks.Task)_selectRewardForPlayer.Invoke(sync, new object[] { set.Player, index })!);
    }

    private static int IndexOfReward(
        MegaCrit.Sts2.Core.Rewards.RewardsSet set, MegaCrit.Sts2.Core.Rewards.Reward reward)
    {
        for (int i = 0; i < set.Rewards.Count; i++)
        {
            if (ReferenceEquals(set.Rewards[i], reward))
            {
                return i;
            }
        }
        throw new InvalidOperationException("Reward not found in the pending set.");
    }

    /// <summary>
    /// Claim one reward from the pending rewards set. For a card reward, the chosen card is
    /// staged on the selector so the game's <c>GetSelectedCardReward</c> seam returns it. The
    /// reward's effects (gain gold/potion/relic, add card to deck) run on the thread pool, so
    /// we pump the selection task and then drain the action queue.
    /// </summary>
    private void TakeReward(GameOption option)
    {
        if (PendingRewards is null)
        {
            throw new InvalidOperationException("No rewards are pending to take.");
        }
        MegaCrit.Sts2.Core.Rewards.Reward reward = option.Reward
            ?? throw new InvalidOperationException("Reward option carried no reward.");

        if (option.CardModel is not null)
        {
            // Card reward: stage which of the offered cards to add to the deck.
            Selector.NextCardRewardPick = option.CardModel;
        }

        SelectRewardForOwner(reward);
        DrainActionQueue();
        // The rewards screen stays up (mirroring the in-game proceed button) even once every
        // reward is taken; the player leaves it explicitly via ProceedFromRewards.
    }

    /// <summary>
    /// Run a card reward's alternative instead of taking a card, reproducing the logic half of the
    /// null <c>NCardRewardSelectionScreen</c>. A <c>DoNothing</c> alternative (reroll) runs its effect
    /// in place and leaves the screen up with the freshly-rolled cards; a terminal alternative
    /// (Pael's Wing sacrifice, …) is staged by id and run through the synchronizer, which marks the
    /// reward selected and runs its effect/hooks — the same path as taking a card.
    /// </summary>
    private void TakeCardRewardAlternative(GameOption option)
    {
        if (PendingRewards is null)
        {
            throw new InvalidOperationException("No rewards are pending to take.");
        }
        MegaCrit.Sts2.Core.Rewards.Reward reward = option.Reward
            ?? throw new InvalidOperationException("Card-reward-alternative option carried no reward.");
        MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative alt =
            option.CardRewardAlternativeModel
            ?? throw new InvalidOperationException("Card-reward-alternative option carried no alternative.");

        if (alt.AfterSelected == MegaCrit.Sts2.Core.Entities.Rewards.PostAlternateCardRewardAction.DoNothing)
        {
            // Reroll-style: the effect (re-roll the offered cards) leaves the reward open, so run it
            // directly rather than through the selection loop, then re-project the new cards.
            Pump(alt.OnSelect());
            DrainActionQueue();
            return;
        }

        // Terminal alternative: stage it by id so the synchronizer's GetSelectedCardReward returns it
        // (the game regenerates the alternative list each round and matches by reference, so a stale
        // instance would not match — the id resolves against the fresh list).
        Selector.NextCardRewardAlternativeId = alt.OptionId;
        SelectRewardForOwner(reward);
        DrainActionQueue();
    }

    /// <summary>
    /// Leave the rewards screen, skipping any rewards not yet taken (mirrors clicking proceed:
    /// the synchronizer marks the remaining rewards skipped and completes the set). For a
    /// post-combat set this returns to the map; for a custom (relic/event) set it resumes the
    /// effect that was suspended offering it, pumping it to quiescence.
    /// </summary>
    private void ProceedFromRewards()
    {
        if (PendingRewards is null)
        {
            throw new InvalidOperationException("Not on a rewards screen to proceed from.");
        }
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        if (!sync.IsRewardsSetCompleted(PendingRewards))
        {
            // Skip the remaining rewards of the current set, routed by its owner.
            if (IsLocalPlayer(PendingRewards.Player))
            {
                sync.SkipLocalRewardsSet();
            }
            else
            {
                _skipRewardsForPlayer.Invoke(sync, new object[] { PendingRewards.Player });
            }
        }
        DrainActionQueue();

        System.Threading.Tasks.TaskCompletionSource? resolve;
        lock (_rewardGate)
        {
            resolve = _customRewardResolve;
            _customRewardResolve = null;
            PendingRewards = null;
            if (resolve is not null)
            {
                _customRewardSignal = NewSignal();
            }
        }

        if (resolve is null)
        {
            // A post-combat set. In a multi-player fight each alive player gets their own rewards,
            // surfaced one at a time — if another player's set is queued, surface it and stay on the
            // rewards screen rather than leaving the room.
            if (_queuedRewardSets.Count > 0)
            {
                PendingRewards = _queuedRewardSets.Dequeue();
                return;
            }

            // Last set done. If this was an act-ending boss room, advance to the next act (or, on the
            // final act, into the Architect victory event) instead of returning to the now-empty map.
            // Otherwise the player is back on the map.
            TryAdvanceActAfterBoss();
            return;
        }

        // Custom set: the offering effect is suspended in RewardsSet.Offer awaiting this; the set
        // is now completed, so unblock it and pump the effect to quiescence (it may finish, raise
        // another choice, or — in combat — continue the action queue).
        resolve.TrySetResult();
        if (_suspendedRoomTask is { IsCompleted: false } roomTask)
        {
            // The suspended effect was a fire-and-forget room/relic task (treasure extra rewards, a
            // relic's on-obtain reward, …); resume it.
            PumpRoomTaskUntilIdleOrChoice(roomTask);
            if (!CustomRewardPending)
            {
                _suspendedRoomTask = null;
            }
        }
        else if (InCombat)
        {
            PumpCombatUntilIdleOrChoice();
        }
        else
        {
            PumpEventUntilIdleOrChoice();
        }
        TryOfferCombatRewards();
    }

    // ---------------------------------------------------------------------------------
    // Act transitions. After an act's boss is beaten and its rewards are dismissed, the in-game
    // rewards screen's proceed button votes to move to the next act (NRewardsScreen →
    // ActChangeSynchronizer.SetLocalPlayerReady → MoveToNextAct → RunManager.EnterNextAct). The
    // harness reproduces the logic half: when the player proceeds from a boss room reached via real
    // map navigation, it drives EnterNextAct directly (single-player has no other voters to wait on,
    // so we skip the fire-and-forget vote action and pump the real transition synchronously, the
    // same way EnterFirstRoom drives EnterAct). EnterNextAct enters the next act's map, or — on the
    // final act — the Architect victory event room.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// If the player just dismissed the rewards of an act-ending boss (a Boss room they reached by
    /// travelling to the boss map node), advance to the next act. A boss entered out of band (via
    /// <see cref="EnterEncounterDebug"/>, not standing on the boss node) just returns to the map, as
    /// does the first boss of a double-boss act (the player travels on to the second boss).
    /// </summary>
    private void TryAdvanceActAfterBoss()
    {
        if (Run.CurrentRoom is not MegaCrit.Sts2.Core.Rooms.CombatRoom room
            || room.RoomType != MegaCrit.Sts2.Core.Rooms.RoomType.Boss)
        {
            return;
        }
        MegaCrit.Sts2.Core.Map.ActMap map = Run.Map;
        if (Run.CurrentMapCoord is not { } coord)
        {
            return; // not standing on a map node (e.g. a debug-entered encounter)
        }
        bool onBossNode = map.BossMapPoint.coord.Equals(coord)
            || (map.SecondBossMapPoint is { } sb && sb.coord.Equals(coord));
        if (!onBossNode)
        {
            return;
        }
        // In a double-boss act, the first boss (at BossMapPoint) does not end the act — the player
        // travels on to the second boss — so only the second boss advances. Mirrors CombatManager's
        // victory check (SecondBossMapPoint == null ? at BossMapPoint : at SecondBossMapPoint).
        bool isFirstOfDoubleBoss =
            map.SecondBossMapPoint is not null && map.BossMapPoint.coord.Equals(coord);
        if (isFirstOfDoubleBoss)
        {
            return;
        }
        AdvanceToNextAct();
    }

    /// <summary>
    /// Drive the transition into the next act (or the Architect victory event on the final act).
    /// Mirrors <c>ActChangeSynchronizer.MoveToNextAct</c>: bump the act-floor counter, then pump the
    /// real <c>RunManager.EnterNextAct</c> to completion. Afterwards the player is on the next act's
    /// map, or in the Architect event room (whose options we initialize like any other event room).
    /// </summary>
    private void AdvanceToNextAct()
    {
        Run.ActFloor++;
        Pump(RunManager.Instance.EnterNextAct());
        // The per-room one-shot guards refer to the act we just left; reset them for the new act.
        _rewardedRoom = null;
        _openedTreasureRoom = null;
        // The final act enters the Architect victory event instead of a map; wait for its options.
        WaitForEventReady();
    }

    /// <summary>
    /// Block until the run reaches its terminal game-over state. Used after the Architect victory
    /// event's proceed votes to win the run: that runs on a fire-and-forget task chain
    /// (EnterNextAct → WinRun → kill all players), so we drain and poll until every player is dead.
    /// </summary>
    private void WaitForGameOver(int timeoutMs = 10000)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (!Run.IsGameOver)
        {
            DrainActionQueue();
            if (Run.IsGameOver)
            {
                break;
            }
            if (sw.ElapsedMilliseconds > timeoutMs)
            {
                throw new System.TimeoutException(
                    "Timed out waiting for the run to end after the Architect victory.");
            }
            System.Threading.Thread.Sleep(5);
        }
    }

    // ---------------------------------------------------------------------------------
    // Events. Choosing an event option runs the option's effect on a thread-pool task via
    // the game's EventSynchronizer; the harness blocks until it finishes (or surfaces a
    // mid-effect card choice). Once the event is finished — or down to only a "proceed"
    // option — the player leaves it by moving on the map (ProceedFromTerminal would drive
    // the null map UI, so we model leaving as a normal MoveTo).
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Choose an event option by its index into the live event's <c>CurrentOptions</c>, then
    /// pump until the option's effect finishes or blocks on a card choice.
    /// </summary>
    private void ApplyEventOption(Player player, int index)
    {
        // The Architect victory event ends the run: its final option votes to move to the next act,
        // which (in the final act's victory room) wins the run and kills all players. That vote — and
        // only that option, not the dialogue advances before it — bumps the act-floor counter (see
        // ActChangeSynchronizer.MoveToNextAct), so a change here flags the run-ending choice.
        int actFloorBefore = Run.ActFloor;

        ChooseEventOptionForPlayer(player, index);
        PumpEventUntilIdleOrChoice();
        // The option may have entered and resolved a combat (shared events); offer its rewards.
        TryOfferCombatRewards();

        // If this option triggered the act-change vote in the Architect's victory room, the win runs
        // on a fire-and-forget task chain (EnterNextAct → WinRun → kill all players); pump until the
        // run reaches its terminal game-over state.
        if (Run.ActFloor != actFloorBefore
            && Run.CurrentRoom is { IsVictoryRoom: true } && !Run.IsGameOver)
        {
            WaitForGameOver();
        }
    }

    // EventSynchronizer.ChooseOptionForEvent(Player, int) is the per-player seam the game's
    // net-message handler invokes when a (remote) player picks an event option; it is private (the
    // public entry, ChooseLocalOption, only ever drives the local player). In a single-process
    // fake-multiplayer run the harness is the input source for *every* player, so for a non-local
    // player it calls that per-player method directly — there is no remote client to send the
    // message that would otherwise reach it. Cached MethodInfo; a signature change fails loudly.
    private static readonly System.Reflection.MethodInfo _chooseOptionForEvent =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer).GetMethod(
            "ChooseOptionForEvent",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(Player), typeof(int) },
            modifiers: null)
        ?? throw new InvalidOperationException(
            "EventSynchronizer.ChooseOptionForEvent(Player, int) not found — the game's event API changed.");

    // For a *shared* (vote-based) event the per-player seam is PlayerVotedForSharedOptionIndex(Player,
    // uint optionIndex, uint pageIndex) — also private, also normally reached via the net-message
    // handler. The current page lives in the private _pageIndex field; we read it for the page arg.
    private static readonly System.Reflection.MethodInfo _playerVotedForShared =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer).GetMethod(
            "PlayerVotedForSharedOptionIndex",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(Player), typeof(uint), typeof(uint) },
            modifiers: null)
        ?? throw new InvalidOperationException(
            "EventSynchronizer.PlayerVotedForSharedOptionIndex(Player, uint, uint) not found — the game's event API changed.");

    private static readonly System.Reflection.FieldInfo _eventPageIndexField =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer).GetField(
            "_pageIndex",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
        ?? throw new InvalidOperationException(
            "EventSynchronizer._pageIndex not found — the game's event API changed.");

    /// <summary>
    /// Choose event option <paramref name="index"/> for the given player. The local player (NetId 1)
    /// uses the faithful <c>ChooseLocalOption</c> path. Any other player in a fake-multiplayer run is
    /// driven through the per-player seam the game's net-message handler would invoke — for a per-player
    /// event that is <c>ChooseOptionForEvent</c>, for a shared (vote-based) event it is
    /// <c>PlayerVotedForSharedOptionIndex</c> (the harness is the input source for every player, so
    /// there is no remote client to send the message). In a shared event the option only *resolves*
    /// once every player has voted; an earlier voter just records their vote.
    /// </summary>
    private void ChooseEventOptionForPlayer(Player player, int index)
    {
        MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer sync = RunManager.Instance.EventSynchronizer;
        bool isLocal = player.NetId == Run.Players[0].NetId;

        if (isLocal)
        {
            // ChooseLocalOption handles both kinds for the local player: it votes for a shared event
            // and runs the option for a per-player one.
            sync.ChooseLocalOption(index);
            return;
        }

        if (sync.GetEventForPlayer(player).IsShared)
        {
            uint page = (uint)_eventPageIndexField.GetValue(sync)!;
            _playerVotedForShared.Invoke(sync, new object[] { player, (uint)index, page });
            return;
        }

        _chooseOptionForEvent.Invoke(sync, new object[] { player, index });
    }

    /// <summary>
    /// Drive an event option's effect to completion. The effect runs as a fire-and-forget
    /// option task (see <c>EventSynchronizer.ChooseOptionForEvent</c>); we await those tasks
    /// but return early if the effect blocks on a card choice, so a blocked choice surfaces
    /// instead of deadlocking. Mirrors <see cref="PumpCombatUntilIdleOrChoice"/>.
    /// </summary>
    private void PumpEventUntilIdleOrChoice(int timeoutMs = 10000)
    {
        System.Threading.Tasks.Task optionTasks =
            RunManager.Instance.EventSynchronizer.AwaitPendingOptionTasks();
        int idx = System.Threading.Tasks.Task.WaitAny(
            new[] { optionTasks, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out resolving an event option (waiting for it to finish, raise a card choice, offer rewards, or open the Crystal Sphere minigame).");
        }
        // If nothing is suspended the option finished; drain anything it enqueued.
        if (!EffectSuspended)
        {
            DrainActionQueue();
        }
    }

    /// <summary>
    /// After entering a room, if it is an event room, wait until the event has been initialized
    /// (options generated, or already finished). <c>EventModel.BeginEvent</c> runs as a
    /// fire-and-forget task, so its options may not be ready the instant room entry returns.
    /// A no-op for non-event rooms.
    /// </summary>
    private void WaitForEventReady(int timeoutMs = 10000)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            DrainActionQueue();
            MegaCrit.Sts2.Core.Models.EventModel? ev = CurrentEvent;
            if (ev is null || ev.IsFinished || ev.CurrentOptions.Count > 0)
            {
                return;
            }
            if (sw.ElapsedMilliseconds > timeoutMs)
            {
                throw new System.TimeoutException("Timed out waiting for the event room to initialize.");
            }
            System.Threading.Thread.Sleep(5);
        }
    }

    // ---------------------------------------------------------------------------------
    // Treasure rooms. Entering one calls TreasureRoomRelicSynchronizer.BeginRelicPicking
    // (populating the relics). The chest-open flow (gold + relic-added extra rewards) and the
    // relic award are normally driven by the null NTreasureRoom / NTreasureRoomRelicCollection
    // UI nodes, so the harness reproduces their logic halves. The relics then surface as
    // GamePhase.Treasure; the agent takes one or skips, after which it leaves via the map.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// If the current room is a freshly-entered treasure room, reproduce the logic half of
    /// <c>NTreasureRoom.OpenChest</c>: grant the chest's gold, then offer any relic-added extra
    /// rewards. Idempotent per room. The relic picking that follows is driven by the agent.
    /// </summary>
    private void TryOpenTreasureChest()
    {
        if (CurrentTreasureRoom is not { } room || ReferenceEquals(room, _openedTreasureRoom))
        {
            return;
        }
        _openedTreasureRoom = room;

        // Chest gold (DoNormalRewards never blocks — it just gains gold).
        _ = Pump(room.DoNormalRewards());
        DrainActionQueue();

        // Extra rewards (normally none): a relic can add reward sets, which go through
        // RewardsSet.Offer → our custom-reward gate. Run as a task and pump until it finishes or
        // suspends on that gate, so a relic-driven reward screen surfaces instead of deadlocking.
        _suspendedRoomTask = room.DoExtraRewardsIfNeeded();
        PumpRoomTaskUntilIdleOrChoice(_suspendedRoomTask);
        if (!CustomRewardPending)
        {
            _suspendedRoomTask = null;
        }

        // A multi-player chest generates one relic per player; on entry BeginRelicPicking *auto-votes*
        // for every non-local player (the fake-multiplayer dummy shortcut). The harness drives those
        // players as real agents, so clear the auto-votes — every player then casts their own pick
        // through the option API (see PickTreasureRelicForPlayer).
        ResetNonLocalTreasureVotes();
    }

    // TreasureRoomRelicSynchronizer._votes is the private per-player vote list; the harness clears the
    // auto-assigned non-local votes so each player picks for real.
    private static readonly System.Reflection.FieldInfo _treasureVotesField =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer).GetField(
            "_votes",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
        ?? throw new InvalidOperationException(
            "TreasureRoomRelicSynchronizer._votes not found — the game's treasure API changed.");

    private void ResetNonLocalTreasureVotes()
    {
        if (Run.Players.Count <= 1 || TreasureSync.CurrentRelics is not { Count: > 0 })
        {
            return;
        }
        var votes = (System.Collections.Generic.List<
            MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer.PlayerVote>)
            _treasureVotesField.GetValue(TreasureSync)!;
        // Slot 0 is the local player (never auto-voted); reset the rest so they pick for themselves.
        for (int i = 1; i < votes.Count; i++)
        {
            votes[i].voteReceived = false;
            votes[i].index = null;
        }
    }

    /// <summary>
    /// Take the treasure relic at <paramref name="index"/>, or skip all relics when null. Mirrors
    /// the logic half of <c>NTreasureRoomRelicCollection</c>: the synchronizer awards relics via
    /// its <c>RelicsAwarded</c> event (consumed here to actually obtain them, since the UI node
    /// that normally does so is null). A singleplayer skip keeps the relics pending until room
    /// exit, so we end the voting explicitly to return the player to the map.
    /// </summary>
    /// <summary>
    /// Cast the given player's treasure pick (a relic index, or null to skip). Single-player drives the
    /// faithful local path (<c>PickRelicLocally</c>/<c>SkipRelicLocally</c>); multi-player records the
    /// pick directly via the synchronizer's <c>OnPicked</c> seam (the path a peer's vote would reach),
    /// resolving and awarding the relics only once every player has picked. Mirrors how the harness
    /// drives non-local players elsewhere (events): it is the input source for all of them.
    /// </summary>
    private void PickTreasureRelicForPlayer(Player player, int? index)
    {
        var sync = TreasureSync;
        System.Collections.Generic.List<MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResult>? results = null;
        void OnAwarded(System.Collections.Generic.List<MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResult> r) => results = r;

        sync.RelicsAwarded += OnAwarded;
        try
        {
            if (Run.Players.Count <= 1)
            {
                if (index is null)
                {
                    sync.SkipRelicLocally();
                }
                else
                {
                    sync.PickRelicLocally(index);
                }
                // Executes the PickRelicAction → OnPicked → AwardRelics → RelicsAwarded.
                DrainActionQueue();
            }
            else
            {
                // Record this player's pick. AwardRelics fires (→ RelicsAwarded) only on the last one.
                sync.OnPicked(player, index);
                DrainActionQueue();
            }
        }
        finally
        {
            sync.RelicsAwarded -= OnAwarded;
        }

        if (results is not null)
        {
            ObtainAwardedRelics(results);
        }

        // A singleplayer skip records the skip but keeps CurrentRelics until the room is exited; end
        // voting now so the player is no longer mid-pick and can move on via the map. (Multi-player
        // resolves through AwardRelics on the final pick, which ends voting itself.)
        if (index is null && Run.Players.Count <= 1)
        {
            sync.OnRoomExited();
        }
    }

    /// <summary>Obtain each relic the synchronizer awarded to a player (the UI node normally does this).</summary>
    private void ObtainAwardedRelics(
        System.Collections.Generic.List<MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResult> results)
    {
        foreach (var result in results)
        {
            if (result.type == MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResultType.Skipped
                || result.player is null)
            {
                continue;
            }
            MegaCrit.Sts2.Core.Models.RelicModel relic = result.relic.ToMutable();
            System.Threading.Tasks.Task obtain = MegaCrit.Sts2.Core.Helpers.TaskHelper.RunSafely(
                MegaCrit.Sts2.Core.Commands.RelicCmd.Obtain(relic, result.player));
            PumpRoomTaskUntilIdleOrChoice(obtain);
        }
    }

    /// <summary>
    /// Drive a fire-and-forget room/effect task until it finishes or suspends on the agent (a
    /// card choice or a custom reward screen), draining the action queue when nothing is
    /// suspended. Mirrors <see cref="PumpCombatUntilIdleOrChoice"/> for out-of-combat tasks.
    /// </summary>
    private void PumpRoomTaskUntilIdleOrChoice(System.Threading.Tasks.Task roomTask, int timeoutMs = 10000)
    {
        int idx = System.Threading.Tasks.Task.WaitAny(
            new[] { roomTask, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out pumping a room task (waiting for it to finish, a card choice, rewards, or the Crystal Sphere minigame).");
        }
        if (!EffectSuspended)
        {
            DrainActionQueue();
        }
    }

    // ---------------------------------------------------------------------------------
    // Rest sites. Entering a RestSiteRoom calls RestSiteSynchronizer.BeginRestSite (generating the
    // rest/smith options). Unlike treasure, the action resolves directly through the synchronizer —
    // ChooseLocalOption runs the option's effect (heal, or Smith's deck upgrade) — so there is no
    // UI logic-half to reproduce. Smith raises a card choice through the same selector seam as
    // combat; Heal's (usually empty) reward offer goes through the custom-reward gate. After a
    // successful choice the game clears the remaining options, leaving the player on the map.
    // ---------------------------------------------------------------------------------

    // RestSiteSynchronizer.ChooseOption(Player, int) is the per-player seam the net-message handler
    // invokes; private (the public ChooseLocalOption only drives the local player). The harness drives
    // each player, so a non-local player goes through it directly (returns the option's effect Task).
    private static readonly System.Reflection.MethodInfo _chooseRestOption =
        typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer).GetMethod(
            "ChooseOption",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(Player), typeof(int) },
            modifiers: null)
        ?? throw new InvalidOperationException(
            "RestSiteSynchronizer.ChooseOption(Player, int) not found — the game's rest API changed.");

    /// <summary>
    /// Choose the rest-site option at <paramref name="index"/> for the given player (into that player's
    /// live option list). The local player uses the faithful <c>ChooseLocalOption</c>; any other player
    /// in a multi-player run goes through the per-player <c>ChooseOption</c> seam. The option's effect
    /// runs as a task; pump it until it finishes or suspends on a card choice (Smith) so the choice
    /// surfaces instead of deadlocking.
    /// </summary>
    private void ChooseRestOption(Player player, int index)
    {
        // The forge ("SMITH") suspends on a deck card choice; remember it so the surfaced choice can be
        // shown as an upgrade preview (the cards the player picks from become their upgraded form).
        bool isForge = index >= 0 && index < RestSiteSync.GetOptionsForPlayer(player.NetId).Count
            && RestSiteSync.GetOptionsForPlayer(player.NetId)[index].OptionId == "SMITH";

        _restChoiceTask = player.NetId == Run.Players[0].NetId
            ? RestSiteSync.ChooseLocalOption(index)
            : (System.Threading.Tasks.Task)_chooseRestOption.Invoke(RestSiteSync, new object[] { player, index })!;
        PumpRoomTaskUntilIdleOrChoice(_restChoiceTask);
        if (isForge && Selector.Pending is { } choice)
        {
            choice.IsUpgradeSelection = true;
        }
        if (_restChoiceTask.IsCompleted)
        {
            _restChoiceTask = null;
        }
    }

    // ---------------------------------------------------------------------------------
    // Merchant shops. Entering a MerchantRoom builds a MerchantInventory per player (cards/relics/
    // potions + a card-removal service). There is no synchronizer or UI logic-half to reproduce on
    // entry; the inventory just exists. Buying runs the entry's faithful purchase path
    // (MerchantEntry.OnTryPurchaseWrapper), which pays gold and grants the item through the same
    // commands as rewards (CardPileCmd.Add / RelicCmd.Obtain / PotionCmd.TryToProcure). The
    // card-removal service raises a deck card choice through the same selector seam as combat/Smith.
    // A relic like The Courier restocks slots after purchase (Hook.ShouldRefillMerchantEntry) and
    // discounts prices (Hook.ModifyMerchantPrice) — both handled by the game logic. The player
    // leaves by moving on the map.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Buy the item the given option points at. Card/relic/potion purchases run the entry's purchase
    /// task to quiescence; the card-removal service suspends on a deck card choice (resolved via
    /// <see cref="Apply"/>'s <see cref="OptionKind.SelectCards"/> path, after which the entry is
    /// marked used since the UI node that would normally do so is null headless).
    /// </summary>
    private void BuyShopItem(GameOption option)
    {
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry = option.ShopEntry
            ?? throw new InvalidOperationException("Buy option carried no shop entry.");
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory inv = InventoryFor(option.Player!);

        if (entry is MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardRemovalEntry removal)
        {
            _shopRemovalEntry = removal;
            _shopRemovalTask = removal.OnTryPurchaseWrapper(inv);
            PumpRoomTaskUntilIdleOrChoice(_shopRemovalTask);
            FinishShopRemovalIfDone();
            return;
        }

        System.Threading.Tasks.Task<bool> task = entry.OnTryPurchaseWrapper(inv);
        PumpRoomTaskUntilIdleOrChoice(task);
        // Card/relic/potion purchases resolve through the action queue (no card choice), so the task
        // is complete here; surface any failure rather than silently swallowing it.
        if (task.IsCompleted && !task.GetAwaiter().GetResult())
        {
            throw new InvalidOperationException($"Shop purchase was rejected: {option.Description}");
        }
    }

    /// <summary>
    /// If the in-flight card-removal purchase has completed, clear the tracking state and — on a
    /// successful removal — mark the entry used (single-use per shop), reproducing the logic half of
    /// <c>NMerchantCardRemoval.OnCardRemovalUsed</c>.
    /// </summary>
    private void FinishShopRemovalIfDone()
    {
        if (_shopRemovalTask is not { IsCompleted: true } task)
        {
            return;
        }
        if (task.GetAwaiter().GetResult())
        {
            _shopRemovalEntry?.SetUsed();
        }
        _shopRemovalTask = null;
        _shopRemovalEntry = null;
    }

    // ---------------------------------------------------------------------------------
    // Crystal Sphere event minigame. The game drives it through a UI screen
    // (NCrystalSphereScreen) that is null headless; a Harmony patch on its ShowScreen captures the
    // plain-C# CrystalSphereMinigame here (OnCrystalSphereScreenShown) instead, suspending the
    // offering event-option task inside PlayMinigame on the minigame's completion source. The agent
    // spends each divination by clearing cells; when the last is spent the minigame completes,
    // resuming PlayMinigame, which grants the fully-revealed items' rewards via RewardsCmd.OfferCustom
    // — the same custom-reward gate as relics/events — and finishes the event.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Spend one divination clearing the grid cell at (<paramref name="x"/>, <paramref name="y"/>)
    /// with the active tool (Big clears the surrounding 3×3 area, Small just the cell). When the last
    /// divination is spent the minigame completes; the revealed items' rewards then surface through
    /// the custom-reward screen (or, if nothing was revealed, the event simply finishes).
    /// </summary>
    private void ClickCrystalSphereCell(int x, int y)
    {
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame =
            PendingCrystalSphere ?? throw new InvalidOperationException("No Crystal Sphere minigame is pending.");

        // CellClicked decrements the divination count, clears the cell(s), and reveals any item whose
        // footprint is now fully uncovered; on the final divination it completes the minigame's
        // completion source, resuming the suspended PlayMinigame task.
        Pump(minigame.CellClicked(minigame.cells[x, y]));

        if (!minigame.IsFinished)
        {
            DrainActionQueue();
            return;
        }

        // Last divination spent: the minigame is over. Clear our pending state, then let the resumed
        // PlayMinigame settle — it grants rewards via the custom-reward gate (which surfaces as a
        // reward screen) and then finishes the event. PumpEventUntilIdleOrChoice returns as soon as a
        // reward screen / card choice surfaces or the event-option task completes.
        lock (_crystalGate)
        {
            _pendingCrystalSphere = null;
            _crystalSphereSignal = NewSignal();
        }
        if (!EffectSuspended)
        {
            PumpEventUntilIdleOrChoice();
        }
    }

    /// <summary>Switch the Crystal Sphere divination tool (Big = 3×3 area, Small = single cell).</summary>
    private void SetCrystalSphereTool(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType tool)
    {
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame =
            PendingCrystalSphere ?? throw new InvalidOperationException("No Crystal Sphere minigame is pending.");
        minigame.SetTool(tool);
    }

    /// <summary>
    /// The live <see cref="Player"/> with the given net id (1-based; 1 is the local player). Throws
    /// if no such player is in the run. Useful for agents driving a specific player in a multi-player
    /// (fake-multiplayer) run. Treat the returned player as read-only from outside the harness.
    /// </summary>
    public Player GetPlayerById(ulong playerId) => GetPlayer(playerId);

    private Player GetPlayer(ulong playerId) =>
        Run.Players.FirstOrDefault(p => p.NetId == playerId)
            ?? throw new ArgumentException($"No player with net id {playerId} in this run.", nameof(playerId));

    private enum CombatPumpResult
    {
        /// <summary>The action queue drained; nothing more to resolve.</summary>
        Idle,

        /// <summary>The current effect is blocked on a card choice (see <see cref="Selector"/>).</summary>
        ChoicePending,
    }

    /// <summary>
    /// Drive the action queue until it either drains or the running effect blocks on a card
    /// choice. The effect runs on a thread-pool continuation, so we wait on whichever happens
    /// first: the queue-drained task or the selector's pending-choice signal.
    /// </summary>
    private CombatPumpResult PumpCombatUntilIdleOrChoice(int timeoutMs = 10000)
    {
        System.Threading.Tasks.Task drain = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
        int idx = System.Threading.Tasks.Task.WaitAny(
            new[] { drain, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out pumping the action queue (waiting for the queue to drain, a card choice, or rewards).");
        }
        // If the queue drained there is no suspended effect; otherwise a choice/reward is pending.
        return EffectSuspended ? CombatPumpResult.ChoicePending : CombatPumpResult.Idle;
    }

    /// <summary>Drive the game's action executor until its queue is empty.</summary>
    private static void DrainActionQueue() =>
        Pump(RunManager.Instance.ActionExecutor.FinishedExecutingActions());

    /// <summary>
    /// Block until the player is back in their Play phase, combat has ended, or the enemy turn
    /// raised a mid-effect choice the agent must resolve.
    ///
    /// The enemy turn resolves on background tasks (it genuinely yields off this thread), so rather
    /// than poll we await a completion source wired to the combat's own events — it fires the instant
    /// the player can act again. We *also* wake on the effect-suspended signals (a card choice / custom
    /// reward / Crystal Sphere): a few enemies raise a player choice on their own turn (e.g.
    /// KnowledgeDemon's curse selection), which would otherwise deadlock this wait — the enemy task is
    /// blocked on the choice, so no turn-started/phase-change ever fires. When that happens the choice
    /// surfaces (<see cref="GamePhase.Choice"/>); resolving it resumes the enemy turn and the caller
    /// waits again. The timeout is a safety net and throws if hit.
    /// </summary>
    private void WaitUntilPlayerCanActOrCombatEnds(Player player, int timeoutMs = 5000)
    {
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        var tcs = new System.Threading.Tasks.TaskCompletionSource(
            System.Threading.Tasks.TaskCreationOptions.RunContinuationsAsynchronously);

        void TryComplete()
        {
            if (PlayerCanActOrCombatEnded(cm, player))
            {
                tcs.TrySetResult();
            }
        }

        void OnTurnStarted(MegaCrit.Sts2.Core.Combat.CombatState _) => TryComplete();
        void OnCombatEnded(MegaCrit.Sts2.Core.Rooms.CombatRoom _) => tcs.TrySetResult();

        // Subscribe before the first check so we cannot miss the transition.
        cm.TurnStarted += OnTurnStarted;
        cm.CombatEnded += OnCombatEnded;
        MegaCrit.Sts2.Core.Entities.Players.PlayerCombatState? pcs = player.PlayerCombatState;
        if (pcs is not null)
        {
            pcs.PlayerTurnPhaseChanged += TryComplete;
        }
        try
        {
            TryComplete(); // in case it is already satisfied
            int idx = System.Threading.Tasks.Task.WaitAny(
                new[] { tcs.Task, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal },
                timeoutMs);
            if (idx < 0)
            {
                throw new System.TimeoutException(
                    "Timed out waiting for the player's turn to resume or combat to end.");
            }
        }
        finally
        {
            cm.TurnStarted -= OnTurnStarted;
            cm.CombatEnded -= OnCombatEnded;
            if (pcs is not null)
            {
                pcs.PlayerTurnPhaseChanged -= TryComplete;
            }
        }

        // A surfaced choice leaves the enemy task suspended; don't drain (it would block on the
        // unresolved choice). Otherwise the turn settled, so drain anything it enqueued.
        if (!EffectSuspended)
        {
            DrainActionQueue();
        }
    }

    private static bool PlayerCanActOrCombatEnded(MegaCrit.Sts2.Core.Combat.CombatManager cm, Player player)
    {
        if (!cm.IsInProgress)
        {
            return true;
        }
        var pcs = player.PlayerCombatState;
        return pcs is not null && pcs.Phase == MegaCrit.Sts2.Core.Combat.PlayerTurnPhase.Play;
    }

    /// <summary>
    /// Drive a game <see cref="Task"/> to completion on the calling thread. The
    /// engine is async/Task-based but, in headless no-delay mode, work either runs
    /// synchronously or completes on the thread pool, so blocking here is safe.
    /// This is a placeholder for the richer quiescence pump.
    /// </summary>
    private static void Pump(System.Threading.Tasks.Task task) => task.GetAwaiter().GetResult();
}
