using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
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

        SaveManager save = SaveManager.Instance;
        UnlockState unlock = save.GenerateUnlockStateFromProgress();

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

        return new GameHost { Run = runState, Seed = seed };
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
    public void MoveTo(MegaCrit.Sts2.Core.Map.MapCoord coord) =>
        Pump(RunManager.Instance.EnterMapCoord(coord));

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
        DrainActionQueue();
        return enqueued;
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
    }

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
            default:
                throw new ArgumentOutOfRangeException(nameof(option), option.Kind, "Unknown option kind.");
        }
    }

    private Player GetPlayer(ulong playerId) =>
        Run.Players.FirstOrDefault(p => p.NetId == playerId)
            ?? throw new ArgumentException($"No player with net id {playerId} in this run.", nameof(playerId));

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
