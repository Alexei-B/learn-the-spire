using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Map;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Every act opens with an Ancient event (Neow in Act 1) whose reward is the act's first stop: the
/// act's starting map node is an Ancient, and with nothing visited yet it is the only travelable
/// node. This verifies that on reaching Act 2 the ancient is offered and grants its event, mirroring
/// <c>NMapScreen.RecalculateTravelability</c>.
/// </summary>
public sealed class AncientTests
{
    private readonly ITestOutputHelper _out;

    public AncientTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task ReachingAct2_OffersTheAncientAsTheOnlyFirstStop()
    {
        await Task.Run(RunToAct2).WaitAsync(TimeSpan.FromSeconds(240));
    }

    private void RunToAct2()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        // Buff HP so the greedy player survives the act-1 boss and reaches act 2.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        GameState atAct2 = AutoPlayer.Advance(
            host,
            stop: s => s.ActIndex == 1 && s.Phase == GamePhase.Map,
            preferMapPointType: MapPointType.Boss,
            maxSteps: 6000,
            log: _out);

        Assert.Equal(1, atAct2.ActIndex);
        Assert.Equal(GamePhase.Map, atAct2.Phase);

        // At the very start of act 2 the only reachable node is the act's Ancient.
        var moves = host.ListOptions().Where(o => o.Kind == OptionKind.MoveTo).ToList();
        Assert.Single(moves);
        Coord ancientCoord = moves[0].Coord!.Value;
        MapPointView node = atAct2.Map!.Points.First(p => p.Coord.Equals(ancientCoord));
        _out.WriteLine($"act2 first stop: ({ancientCoord.Col},{ancientCoord.Row}) {node.PointType}");
        Assert.Equal(MapPointType.Ancient, node.PointType);

        // Entering it presents the act's ancient event (a reward choice), not a combat.
        host.Apply(moves[0]);
        GameState atAncient = host.GetState();
        _out.WriteLine($"after entering ancient: phase={atAncient.Phase} event={atAncient.Event?.EventId}");
        Assert.Equal(GamePhase.Event, atAncient.Phase);
        Assert.NotNull(atAncient.Event);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.ChooseEventOption);
    }
}
