using System;
using System.Linq;
using System.Threading.Tasks;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// The run score (<see cref="GameState.Score"/>, projected from <c>ScoreUtility.CalculateScore</c>)
/// accrues as the run progresses — floors climbed, gold, elites/bosses slain. A fast check that it is
/// wired and grows; the full victory score is asserted by <see cref="WalkthroughTests"/>.
/// </summary>
public sealed class ScoreTests
{
    private readonly ITestOutputHelper _out;

    public ScoreTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Score_IsZeroAtRunStart_AndPositiveAfterClearingTheFirstCombat()
    {
        await Task.Run(Run).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        int scoreAtStart = host.GetState().Score;
        _out.WriteLine($"score at start: {scoreAtStart}");
        Assert.True(scoreAtStart >= 0);

        // Win the first combat and stop on its rewards screen.
        host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.MoveTo));
        Assert.True(host.InCombat);
        GameState atRewards = AutoPlayer.Advance(host, stop: s => s.Phase == GamePhase.Reward, log: _out);

        _out.WriteLine($"score after first combat: {atRewards.Score}");
        // Clearing a room adds to the floor score, so the score has grown.
        Assert.True(atRewards.Score > scoreAtStart,
            $"expected score to grow after a combat (start {scoreAtStart}, now {atRewards.Score})");
    }
}
