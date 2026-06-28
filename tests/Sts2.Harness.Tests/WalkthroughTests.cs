using System;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// End-to-end "play a real game forward" test: from the opening Neow event, a greedy
/// <see cref="AutoPlayer"/> drives the run through the whole of act 1 — events, combats (faithful
/// card play with block-then-attack, enemy turns), post-combat rewards, rest sites, treasure, shops,
/// and map navigation — entirely through the public option API, until it **beats the act-1 boss or
/// dies**. The player is buffed to a huge HP pool so the still-simple greedy combat survives, so on
/// the standard seed it reaches and defeats the boss (CeremonialBeast at floor 17). Beating the boss
/// leaves the player at the end of the act with no further map moves (act 1 → 2 transition is M4 and
/// not built yet), which surfaces as <see cref="GamePhase.Other"/> with no reachable points — the
/// terminal "won the act" state this test asserts.
/// </summary>
public sealed class WalkthroughTests
{
    private readonly ITestOutputHelper _out;

    public WalkthroughTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task GreedyRun_PlaysThroughAct1_UntilItBeatsTheBossOrDies()
    {
        var t = Task.Run(Run);
        await t.WaitAsync(TimeSpan.FromSeconds(180));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Buff the player to a huge HP pool so the (still simple) greedy combat survives the act and
        // we exercise the full room/navigation breadth rather than dying to chip damage.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        // Play forward, steering toward the boss, until the run ends: either the player dies
        // (GameOver) or the boss is beaten — after which act 1 has no more reachable rooms (the act
        // transition is M4), so the run sits on GamePhase.Other with nowhere left to move.
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.IsGameOver
                || (s.Phase == GamePhase.Other && (s.Map?.Reachable.Count ?? 0) == 0),
            preferMapPointType: MegaCrit.Sts2.Core.Map.MapPointType.Boss,
            log: _out);

        _out.WriteLine($"Run ended: phase={end.Phase} act={end.ActIndex} floor={end.Floor} gameover={end.IsGameOver} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp} relics=[{string.Join(",", end.Players[0].Relics)}]");

        // With the HP buff the greedy player wins, so the run ends alive at the end of act 1: the
        // boss (the act's only terminal node) has been cleared, leaving no reachable map moves.
        Assert.False(end.IsGameOver, "expected the buffed greedy player to survive the act");
        Assert.True(end.Players[0].CurrentHp > 0);
        Assert.Equal(0, end.ActIndex);
        Assert.Equal(GamePhase.Other, end.Phase);
        Assert.Equal(0, end.Map?.Reachable.Count ?? 0);
        // Reaching the boss means a deep run: ~17 floors of combats/events/rest/treasure.
        Assert.True(end.Floor >= 16, $"expected to reach the boss (floor ~17) but stopped on floor {end.Floor}");
        // Evidence of treasure-room traversal: the starting + Neow relic plus a treasure relic.
        Assert.True(end.Players[0].Relics.Count >= 3,
            $"expected to have picked up a treasure relic but have only {end.Players[0].Relics.Count} relics");
    }
}
