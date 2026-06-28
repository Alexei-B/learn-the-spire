using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace Sts2.Harness;

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
    /// The post-combat rewards currently on offer, or null when not on the rewards screen.
    /// Surfaced as <see cref="GamePhase.Reward"/> and resolved through the reward options.
    /// </summary>
    internal MegaCrit.Sts2.Core.Rewards.RewardsSet? PendingRewards { get; private set; }

    // The combat room whose rewards we have already generated, so we offer them exactly once
    // per fight even while the player lingers on the rewards screen before moving on.
    private MegaCrit.Sts2.Core.Rooms.AbstractRoom? _rewardedRoom;

    /// <summary>
    /// Create and start a fresh single-player run with the given seed.
    /// Mirrors the game's own NSceneBootstrapper flow, minus all UI/asset steps.
    /// </summary>
    public static GameHost StartNewRun(string seed, int ascension = 0)
    {
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

        CharacterModel character = ModelDb.AllCharacters.First(); // Ironclad
        Player player = Player.CreateForNewRun(character, unlock, 1uL);

        List<ActModel> acts = ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList();
        RunState runState = RunState.CreateForNewRun(
            new List<Player> { player },
            acts,
            new List<ModifierModel>(),
            GameMode.Standard,
            ascension,
            seed);

        RunManager.Instance.SetUpNewSingleplayer(runState, shouldSave: false);
        RunManager.Instance.Launch();

        // Install our card selector so mid-effect choices route to the harness. Disposing the
        // previous scope clears the process-wide selector stack before we claim it.
        _selectorScope?.Dispose();
        var selector = new HarnessCardSelector();
        _selectorScope = MegaCrit.Sts2.Core.Commands.CardSelectCmd.UseSelector(selector);

        return new GameHost { Run = runState, Seed = seed, Selector = selector };
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
    }

    /// <summary>
    /// The local player's mutable event when the current room is an (out-of-combat) event room,
    /// or null otherwise. Used to surface <see cref="GamePhase.Event"/> options.
    /// </summary>
    internal MegaCrit.Sts2.Core.Models.EventModel? CurrentEvent =>
        RunManager.Instance.IsInProgress
        && !InCombat
        && Run.CurrentRoom is MegaCrit.Sts2.Core.Rooms.EventRoom
            ? RunManager.Instance.EventSynchronizer.GetLocalEvent()
            : null;

    /// <summary>
    /// True when the current event is awaiting a real (non-proceed, unlocked) choice from the
    /// player. A finished event — or one whose only remaining option is "proceed" — is not
    /// actionable: the player leaves it by moving on the map.
    /// </summary>
    internal bool HasActionableEvent =>
        CurrentEvent is { } e && e.CurrentOptions.Any(o => !o.IsLocked && !o.IsProceed);

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
    /// End the given player's turn and resolve the enemy turn to quiescence: the enemy
    /// turn runs on background tasks (not the player action queue), so we wait (via the
    /// combat's events) until the player can act again or combat ends.
    /// </summary>
    public void EndTurn(Player player)
    {
        MegaCrit.Sts2.Core.Commands.PlayerCmd.EndTurn(player, canBackOut: false);
        DrainActionQueue();
        WaitUntilPlayerCanActOrCombatEnds(player);
        TryOfferCombatRewards();
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
        // Only the victor gets rewards; a dead player means a loss (handled as game over).
        if (!Run.Players[0].Creature.IsAlive)
        {
            return;
        }
        if (room.Encounter is { ShouldGiveRewards: false })
        {
            return;
        }

        _rewardedRoom = room;
        Player player = Run.Players[0];

        // GenerateForRoomEnd populates the rewards and runs reward-modifying hooks but does not
        // offer them; we then register the set so the synchronizer's select/skip APIs work.
        MegaCrit.Sts2.Core.Rewards.RewardsSet set =
            Pump(MegaCrit.Sts2.Core.Commands.RewardsCmd.GenerateForRoomEnd(player, room));
        _ = RunManager.Instance.RewardsSetSynchronizer.BeginRewardsSet(set);
        PendingRewards = set;
    }

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

        // The post-combat rewards screen: take rewards then proceed back to the map.
        if (PendingRewards is not null)
        {
            return BuildRewardOptions(player, PendingRewards);
        }

        // An event room awaiting a choice (e.g. the opening Neow ancient event).
        if (HasActionableEvent)
        {
            return BuildEventOptions(player, CurrentEvent!);
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
                CardView view = GameStateProjection.ProjectCard(card, canPlay: true);
                if (card.TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                {
                    foreach (MegaCrit.Sts2.Core.Entities.Creatures.Creature enemy in combat.HittableEnemies)
                    {
                        if (card.IsValidTarget(enemy))
                        {
                            options.Add(GameOption.PlayCardOption(player, card, view, enemy));
                        }
                    }
                }
                else
                {
                    options.Add(GameOption.PlayCardOption(player, card, view, target: null));
                }
            }

            options.Add(GameOption.EndTurnOption(player));
            return options;
        }

        // Out of combat: map room choices.
        foreach (MegaCrit.Sts2.Core.Map.MapPoint point in GameStateProjection.ReachablePoints(Run))
        {
            options.Add(GameOption.MoveToOption(player, point.coord));
        }
        return options;
    }

    /// <summary>List options for the first (or only) player. Convenience for single-player.</summary>
    public IReadOnlyList<GameOption> ListOptions() => ListOptions(Run.Players[0].NetId);

    /// <summary>
    /// Build the options that resolve a pending card choice. Single-select choices enumerate
    /// one option per card (plus a skip when the choice allows selecting none). Multi-select
    /// choices (min &gt; 1) currently offer a single exact-minimum selection so play never
    /// blocks; full subset enumeration is future work.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildChoiceOptions(Player player, PendingChoice pending)
    {
        var options = new List<GameOption>();
        var views = pending.Options
            .Select(c => GameStateProjection.ProjectCard(c, canPlay: false))
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
    private static IReadOnlyList<GameOption> BuildRewardOptions(Player player, MegaCrit.Sts2.Core.Rewards.RewardsSet set)
    {
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
                MoveTo(option.MapCoord!.Value);
                break;
            case OptionKind.SelectCards:
                if (Selector.Pending is null)
                {
                    throw new InvalidOperationException("No card choice is pending to resolve.");
                }
                Selector.Resolve(option.SelectedCardModels!);
                // The effect resumes on the thread pool; pump until it finishes or blocks again.
                // A choice raised mid-event resumes an event-option task rather than the combat
                // action queue, so pump the matching surface.
                if (InCombat)
                {
                    PumpCombatUntilIdleOrChoice();
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
            case OptionKind.ProceedFromRewards:
                ProceedFromRewards();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(option), option.Kind, "Unknown option kind.");
        }
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

        Pump(RunManager.Instance.RewardsSetSynchronizer.SelectLocalReward(reward));
        DrainActionQueue();
        // The rewards screen stays up (mirroring the in-game proceed button) even once every
        // reward is taken; the player leaves it explicitly via ProceedFromRewards.
    }

    /// <summary>
    /// Leave the rewards screen, skipping any rewards not yet taken (mirrors clicking proceed:
    /// the synchronizer marks the remaining rewards skipped and completes the set), and return
    /// to the map.
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
            sync.SkipLocalRewardsSet();
        }
        DrainActionQueue();
        PendingRewards = null;
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
        RunManager.Instance.EventSynchronizer.ChooseLocalOption(index);
        PumpEventUntilIdleOrChoice();
        // The option may have entered and resolved a combat (shared events); offer its rewards.
        TryOfferCombatRewards();
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
        System.Threading.Tasks.Task pending = Selector.PendingSignal;
        int idx = System.Threading.Tasks.Task.WaitAny(new[] { optionTasks, pending }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out resolving an event option (waiting for it to finish or raise a card choice).");
        }
        // If no choice is pending the option finished; drain anything it enqueued.
        if (Selector.Pending is null)
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
        System.Threading.Tasks.Task pending = Selector.PendingSignal;
        int idx = System.Threading.Tasks.Task.WaitAny(new[] { drain, pending }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out pumping the action queue (waiting for the queue to drain or a card choice).");
        }
        // If the queue drained there is no suspended effect, so no choice can be pending.
        return Selector.Pending is not null ? CombatPumpResult.ChoicePending : CombatPumpResult.Idle;
    }

    /// <summary>Drive the game's action executor until its queue is empty.</summary>
    private static void DrainActionQueue() =>
        Pump(RunManager.Instance.ActionExecutor.FinishedExecutingActions());

    /// <summary>
    /// Block until the player is back in their Play phase or combat has ended.
    ///
    /// The enemy turn resolves on background tasks (it genuinely yields off this
    /// thread), so rather than poll we await a completion source wired to the combat's
    /// own events — it fires the instant the player can act again. The timeout is only
    /// a safety net against a hang and throws if hit (it should never be reached).
    /// </summary>
    private static void WaitUntilPlayerCanActOrCombatEnds(Player player, int timeoutMs = 5000)
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
            if (!tcs.Task.Wait(timeoutMs))
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

        DrainActionQueue();
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
