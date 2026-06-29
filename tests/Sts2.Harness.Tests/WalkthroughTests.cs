using System;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// End-to-end "play a real game forward" test: from the opening Neow event, a greedy
/// <see cref="AutoPlayer"/> drives the run through all three acts — events, combats (faithful card
/// play with block-then-attack, enemy turns), post-combat rewards, rest sites, treasure, shops, map
/// navigation, the act 1→2→3 boss handoffs, and finally the Architect victory event — entirely
/// through the public option API, until it **wins the run or dies**. The player is buffed to a huge
/// HP pool so the still-simple greedy combat survives, so on the standard seed it beats all three
/// act bosses and wins, ending on <see cref="GamePhase.GameOver"/> with <see cref="GameState.IsVictory"/>.
/// </summary>
public sealed class WalkthroughTests
{
    private readonly ITestOutputHelper _out;

    public WalkthroughTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task GreedyRun_PlaysThroughAllThreeActs_AndWinsTheRun()
    {
        var t = Task.Run(Run);
        await t.WaitAsync(TimeSpan.FromSeconds(300));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Buff the player to a huge HP pool so the (still simple) greedy combat survives the run and
        // we exercise the full room/navigation/act-transition breadth rather than dying to chip
        // damage. Energy/deck are untouched, so fights still resolve through real mechanics.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        // Play forward, always steering toward the boss, until the run ends — either the player dies
        // (a game-over with no victory) or every act boss falls and the Architect event wins the run
        // (a game-over flagged IsVictory, since the game ends a victory by killing the players).
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.IsGameOver,
            preferMapPointType: MegaCrit.Sts2.Core.Map.MapPointType.Boss,
            log: _out);

        _out.WriteLine($"Run ended: phase={end.Phase} act={end.ActIndex} floor={end.Floor} victory={end.IsVictory} score={end.Score} relics=[{string.Join(",", end.Players[0].Relics)}]");

        // With the HP buff the greedy player wins: it cleared all three acts and the Architect.
        Assert.True(end.IsGameOver);
        Assert.True(end.IsVictory, "expected the buffed greedy player to win the run, not die");
        // Winning means reaching act 3 (index 2) and clearing its boss.
        Assert.Equal(2, end.ActIndex);
        // A full run traverses ~45 floors of combats/events/rest/treasure/shops across three acts.
        Assert.True(end.Floor >= 30, $"expected a deep three-act run but stopped on floor {end.Floor}");
        // Evidence of treasure-room traversal across acts: the starting + Neow relic plus more.
        Assert.True(end.Players[0].Relics.Count >= 3,
            $"expected to have picked up relics across the run but have only {end.Players[0].Relics.Count}");
        // A full winning run scores well into the thousands (floors × act + elites/bosses + gold).
        Assert.True(end.Score > 0, $"expected a positive run score but got {end.Score}");
    }
}
