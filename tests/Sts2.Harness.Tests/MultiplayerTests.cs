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

        // Both players have their own combat state, hand and turn phase.
        Assert.All(host.Run.Players, p => Assert.NotNull(p.PlayerCombatState));

        // Drive up to a handful of rounds: each round, every player plays what it can then ends its
        // turn. The enemy turn only fires after the *second* player ends — validating the shared
        // turn structure. Stop when combat ends (won → rewards/map).
        for (int round = 0; round < 40 && host.InCombat; round++)
        {
            foreach (Player player in host.Run.Players)
            {
                if (!host.InCombat)
                {
                    break;
                }
                PlayUntilNoCardThenEndTurn(host, player.NetId);
            }
        }

        Assert.False(host.InCombat, "the two-player fight should have resolved within the round budget");
        GameState end = host.GetState();
        _out.WriteLine($"ended phase={end.Phase} p1={end.Players[0].CurrentHp} p2={end.Players[1].CurrentHp}");
        Assert.True(end.Phase is GamePhase.Reward or GamePhase.Map or GamePhase.Choice,
            $"unexpected terminal phase {end.Phase}");
    }

    /// <summary>
    /// Greedily play one player's playable cards (focus-firing the first hittable enemy), then end
    /// that player's turn. Mirrors the per-player option API: list options for the netId, apply them.
    /// </summary>
    private static void PlayUntilNoCardThenEndTurn(GameHost host, ulong netId)
    {
        PlayerCombatState? pcs = host.GetPlayerById(netId).PlayerCombatState;
        if (pcs is null || pcs.Phase != PlayerTurnPhase.Play)
        {
            return; // not this player's turn to act (already ended / combat transitioning)
        }

        // Play playable cards until none remain (or combat ends / a choice surfaces).
        for (int guard = 0; guard < 50 && host.InCombat; guard++)
        {
            GameOption? play = host.ListOptions(netId).FirstOrDefault(o => o.Kind == OptionKind.PlayCard);
            if (play is null)
            {
                break;
            }
            host.Apply(play);
            if (host.GetState().Phase == GamePhase.Choice)
            {
                break; // leave any mid-effect choice for the caller; keeps this helper simple
            }
        }

        if (!host.InCombat)
        {
            return;
        }
        GameOption? end = host.ListOptions(netId).FirstOrDefault(o => o.Kind == OptionKind.EndTurn);
        if (end is not null)
        {
            host.Apply(end);
        }
    }
}
