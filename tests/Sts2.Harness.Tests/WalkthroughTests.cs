using System;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// End-to-end "play a real game forward" smoke test: from the opening Neow event, a greedy
/// <see cref="AutoPlayer"/> drives the run through the first act's rooms (events, combats with
/// faithful card play and enemy turns, post-combat rewards, map navigation) entirely through the
/// public option API. The current greedy combat play is not strong enough to clear the act, so the
/// run advances several floors and then dies — exactly the "beat the boss or die" loop the harness
/// is meant to support. This guards that a multi-floor run executes without the harness throwing.
/// </summary>
public sealed class WalkthroughTests
{
    private readonly ITestOutputHelper _out;

    public WalkthroughTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task GreedyRun_AdvancesThroughSeveralFloors_ThenReachesATerminalState()
    {
        var t = Task.Run(Run);
        await t.WaitAsync(TimeSpan.FromSeconds(180));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Play until the run ends or the driver hits a room it cannot model. Either way it should
        // get well into the act without an exception.
        GameState end = AutoPlayer.Advance(host, stop: _ => false, log: _out);

        _out.WriteLine($"Run ended: phase={end.Phase} floor={end.Floor} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp}");

        Assert.True(end.Floor >= 5, $"expected to reach several floors but stopped on floor {end.Floor}");
        // The greedy driver currently dies partway through act 1; a clean game-over (not an
        // exception or a stuck unmodelled room) is the expected terminal state.
        Assert.Equal(GamePhase.GameOver, end.Phase);
        Assert.True(end.IsGameOver);
    }
}
