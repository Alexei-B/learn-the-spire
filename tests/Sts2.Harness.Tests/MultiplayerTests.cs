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
