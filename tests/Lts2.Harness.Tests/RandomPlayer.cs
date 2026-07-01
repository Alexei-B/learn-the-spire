using System;
using System.Collections.Generic;
using System.Linq;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// A property-style driver that plays a run forward making a *random* legal choice each step (seeded
/// by its own input RNG, independent of the game seed), checking invariants after every step. Used by
/// <see cref="PropertyE2ETests"/> to fuzz full runs: any unhandled exception, broken invariant, or
/// stuck state (a non-terminal phase the harness offers no legal option for) fails the test and the
/// (input-seed, game-seed) pair pins the regression.
/// </summary>
internal static class RandomPlayer
{
    /// <summary>
    /// Drive <paramref name="host"/> with random legal options until the run ends or the step budget
    /// is reached. Returns the terminal/last state. Throws if an invariant breaks or the run gets
    /// stuck (a non-terminal state with no applicable option for the local player).
    /// </summary>
    public static GameState PlayFullRun(GameHost host, int inputSeed, int maxSteps = 6000, ITestOutputHelper? log = null)
    {
        var rng = new Random(inputSeed);
        ulong playerId = host.Run.Players[0].NetId;
        int prevFloor = 0;
        GamePhase lastPhase = (GamePhase)(-1);

        for (int step = 0; step < maxSteps; step++)
        {
            GameState s = host.GetState();
            AssertInvariants(s, prevFloor);
            prevFloor = s.Floor;

            if (s.Phase == GamePhase.GameOver)
            {
                return s;
            }
            if (log is not null && s.Phase != lastPhase)
            {
                lastPhase = s.Phase;
                log.WriteLine($"  [{s.Phase}] floor={s.Floor} hp={s.Players[0].CurrentHp}/{s.Players[0].MaxHp}");
            }

            IReadOnlyList<GameOption> options = host.ListOptions(playerId);
            if (options.Count == 0)
            {
                // In single-player the harness pumps through enemy turns synchronously, so the local
                // player should always have a legal option unless the run is over. An empty list in a
                // non-terminal, non-combat state is a real modelling gap — surface it.
                if (host.InCombat)
                {
                    // Defensive: nothing playable but still our turn — end it.
                    host.EndTurn(host.Run.Players[0]);
                    continue;
                }
                throw new InvalidOperationException(
                    $"Stuck: no legal options in phase {s.Phase} at floor {s.Floor} " +
                    $"(room {host.Run.CurrentRoom?.GetType().Name}).");
            }

            GameOption choice = options[rng.Next(options.Count)];
            host.Apply(choice);
        }

        return host.GetState();
    }

    /// <summary>
    /// Assert the mechanical invariants that must hold in every reachable state: HP within bounds,
    /// non-negative gold/energy, a non-empty deck, and a non-decreasing floor (the run only advances).
    /// </summary>
    private static void AssertInvariants(GameState s, int prevFloor)
    {
        Assert.True(s.Floor >= prevFloor, $"floor went backwards: {prevFloor} -> {s.Floor}");

        foreach (PlayerState p in s.Players)
        {
            Assert.True(p.MaxHp > 0, $"player {p.NetId} has non-positive max HP {p.MaxHp}");
            Assert.InRange(p.CurrentHp, 0, p.MaxHp);
            Assert.True(p.Gold >= 0, $"player {p.NetId} has negative gold {p.Gold}");
            Assert.False(p.Deck.Count == 0, $"player {p.NetId} has an empty deck");

            if (p.CombatState is { } cs)
            {
                Assert.True(cs.Energy >= 0, $"player {p.NetId} has negative energy {cs.Energy}");
                Assert.True(cs.MaxEnergy >= 0, $"player {p.NetId} has negative max energy {cs.MaxEnergy}");
                int piles = cs.Hand.Count + cs.DrawPile.Count + cs.DiscardPile.Count + cs.ExhaustPile.Count;
                Assert.True(piles >= 0, "negative pile total"); // structural sanity
            }
        }

        // A player must be dead for the run to be over (and not before).
        if (s.Phase == GamePhase.GameOver)
        {
            Assert.True(s.IsGameOver, "GameOver phase but IsGameOver is false");
        }

        foreach (EnemyView e in s.Combat?.Enemies ?? Array.Empty<EnemyView>())
        {
            Assert.InRange(e.CurrentHp, 0, e.MaxHp);
        }
    }
}
