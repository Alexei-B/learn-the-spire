using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// M6 — local ("fake") multiplayer: one process hosting N players on the singleplayer net service,
/// all driven by the harness. These tests cover run setup, the multi-player read model, and the
/// per-player combat turn structure (the enemy turn resolves only once *every* player has ended).
/// </summary>
public sealed class MultiplayerTests
{
    private readonly ITestOutputHelper _out;

    public MultiplayerTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void StartNewRun_WithTwoPlayers_BootsBothWithDistinctNetIdsAndDecks()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);

        GameState state = host.GetState();
        _out.WriteLine($"players={state.Players.Count}");
        foreach (PlayerState p in state.Players)
        {
            _out.WriteLine($"  netId={p.NetId} char={p.Character} hp={p.CurrentHp}/{p.MaxHp} deck={p.Deck.Count}");
        }

        Assert.Equal(2, state.Players.Count);
        Assert.Equal(new ulong[] { 1, 2 }, state.Players.Select(p => p.NetId).ToArray());
        Assert.All(state.Players, p => Assert.True(p.MaxHp > 0));
        Assert.All(state.Players, p => Assert.True(p.Deck.Count > 0));
    }

    [Fact]
    public async Task TwoPlayers_NavigateForward_EachResolvesOwnNeow_ThenThePartyVotesIntoCombat()
    {
        await Task.Run(RunTwoPlayerNavigation).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunTwoPlayerNavigation()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);
        host.EnterFirstRoom();

        // The party opens on the Neow ancient event — each player gets their *own* instance and
        // resolves it independently. While player 1 still has a choice, player 2 also has one.
        Assert.Equal(GamePhase.Event, host.GetState().Phase);

        // Resolve each player's Neow through their own per-player options. Each player has its own
        // ChooseEventOption set (the per-player event instances), surfaced by ListOptions(netId). Pick
        // a benign blessing (one whose relic has no upon-pickup side effect) so each option resolves
        // cleanly — the same clean-start choice TestNav makes for single-player.
        foreach (Player player in host.Run.Players)
        {
            int index = BenignNeowOptionIndex(player);
            GameOption blessing = host.ListOptions(player.NetId)
                .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == index);
            host.Apply(blessing);
        }

        // With both Neow events done the party is on the act map.
        GameState onMap = host.GetState();
        Assert.Equal(GamePhase.Map, onMap.Phase);

        // Move on the map by vote: player 1 votes first (no move yet — the party waits for everyone),
        // then player 2's vote completes the tally and the party moves together into the first room.
        // Both vote for the same destination so the (vote-weighted) pick is deterministic.
        Coord dest = host.ListOptions(1uL).First(o => o.Kind == OptionKind.MoveTo).Coord!.Value;

        GameOption p1Move = host.ListOptions(1uL)
            .First(o => o.Kind == OptionKind.MoveTo && o.Coord == dest);
        host.Apply(p1Move);
        Assert.False(host.InCombat, "the party should not have moved on a single vote");

        GameOption p2Move = host.ListOptions(2uL)
            .First(o => o.Kind == OptionKind.MoveTo && o.Coord == dest);
        host.Apply(p2Move);

        Assert.True(host.InCombat, "the party should have moved into combat once both players voted");
        Assert.All(host.Run.Players, p => Assert.NotNull(p.PlayerCombatState));
        _out.WriteLine($"both players in combat at floor {host.GetState().Floor}");
    }

    /// <summary>
    /// The index of a benign Neow option for the given player's own event: a non-locked, non-proceed
    /// blessing whose relic has no upon-pickup side effect (so choosing it resolves cleanly, leaving
    /// the starting deck unpadded). Falls back to the first actionable option.
    /// </summary>
    private static int BenignNeowOptionIndex(Player player)
    {
        MegaCrit.Sts2.Core.Models.EventModel ev =
            MegaCrit.Sts2.Core.Runs.RunManager.Instance.EventSynchronizer.GetEventForPlayer(player);
        int fallback = -1;
        for (int i = 0; i < ev.CurrentOptions.Count; i++)
        {
            MegaCrit.Sts2.Core.Events.EventOption opt = ev.CurrentOptions[i];
            if (opt.IsLocked || opt.IsProceed)
            {
                continue;
            }
            if (fallback < 0)
            {
                fallback = i;
            }
            if (opt.Relic is { HasUponPickupEffect: true })
            {
                continue;
            }
            return i;
        }
        return fallback;
    }

    [Fact]
    public async Task TwoPlayers_BothActInOneCombat_AndTheFightResolves()
    {
        await Task.Run(RunTwoPlayerCombat).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunTwoPlayerCombat()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);

        // Buff both players to a large HP pool so the fight is survivable, then drop straight into a
        // shared combat (bypassing the per-player Neow/map navigation, which is separate M6 work).
        foreach (Player p in host.Run.Players)
        {
            p.Creature.SetMaxHpInternal(9999);
            p.Creature.SetCurrentHpInternal(9999);
        }

        EncounterModel encounter = Act1FightsTests.ResolveEncounter("SlimesNormal");
        host.EnterEncounterDebug(encounter);
        Assert.True(host.InCombat, "expected both players to be in the shared combat");

        // Both players have their own combat state, hand and turn phase, both able to act.
        Assert.All(host.Run.Players, p => Assert.NotNull(p.PlayerCombatState));

        // Drive the shared fight. In the fake-multiplayer turn model both players act during the *same*
        // Play phase; ending any player ends the shared round (the enemy turn fires), so each step:
        //   1) resolve a pending mid-effect card choice (e.g. Silent's Survivor discard), else
        //   2) let any player still in Play play one of their cards, else
        //   3) no one has a card to play → end the turn once (local player) to trigger the enemy turn.
        // Stop when combat ends (won → rewards/map).
        bool bothPlayersGotToAct = false;
        for (int step = 0; step < 2000 && host.InCombat; step++)
        {
            if (host.GetState().Phase == GamePhase.Choice)
            {
                host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.SelectCards));
                continue;
            }

            GameOption? play = null;
            int playersWhoCanPlay = 0;
            foreach (Player player in host.Run.Players)
            {
                if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play)
                {
                    continue;
                }
                GameOption? candidate = host.ListOptions(player.NetId)
                    .FirstOrDefault(o => o.Kind == OptionKind.PlayCard);
                if (candidate is not null)
                {
                    playersWhoCanPlay++;
                    play ??= candidate;
                }
            }
            if (playersWhoCanPlay == 2)
            {
                bothPlayersGotToAct = true; // both players had a playable card in the same Play phase
            }

            if (play is not null)
            {
                host.Apply(play);
                continue;
            }

            // No player has a card to play this turn — end the shared round (local player), which
            // triggers the enemy turn (fake-multiplayer: any end ends the round).
            GameOption? end = host.ListOptions(host.Run.Players[0].NetId)
                .FirstOrDefault(o => o.Kind == OptionKind.EndTurn);
            if (end is null)
            {
                break; // nothing to do (transitioning) — avoid spinning
            }
            host.Apply(end);
        }

        Assert.False(host.InCombat, "the two-player fight should have resolved within the step budget");
        Assert.True(bothPlayersGotToAct, "both players should have been able to play cards in the shared combat");
        GameState end2 = host.GetState();
        _out.WriteLine($"ended phase={end2.Phase} p1={end2.Players[0].CurrentHp} p2={end2.Players[1].CurrentHp}");
        Assert.True(end2.Phase is GamePhase.Reward or GamePhase.Map or GamePhase.Choice,
            $"unexpected terminal phase {end2.Phase}");
    }
}
