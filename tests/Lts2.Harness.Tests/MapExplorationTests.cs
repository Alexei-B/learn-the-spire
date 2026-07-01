using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Map;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

public sealed class MapExplorationTests
{
    private readonly ITestOutputHelper _out;

    public MapExplorationTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Map_IsReadable_AndCanMoveIntoFirstRoom()
    {
        // With all epochs unlocked the run opens on the Neow ancient event; resolve it to reach the map.
        GameHost host = TestNav.StartOnMap("TESTSEED");

        var rs = host.Run;
        _out.WriteLine($"CurrentRoom={rs.CurrentRoom?.GetType().Name} curCoord={rs.CurrentMapCoord} act={rs.CurrentActIndex}");
        _out.WriteLine($"Map type={rs.Map.GetType().Name} startingPoint={rs.Map.StartingMapPoint.coord} ({rs.Map.StartingMapPoint.PointType})");

        MapPoint? cur = rs.CurrentMapPoint;
        IEnumerable<MapPoint> next = cur != null ? cur.Children : rs.Map.StartingMapPoint.Children;
        List<MapPoint> options = next.OrderBy(p => p.coord.col).ToList();

        _out.WriteLine($"Selectable next points ({options.Count}):");
        foreach (MapPoint p in options)
        {
            _out.WriteLine($"  ({p.coord.col},{p.coord.row}) {p.PointType}");
        }

        Assert.NotEmpty(options);

        MapPoint chosen = options[0];
        _out.WriteLine($"Moving to ({chosen.coord.col},{chosen.coord.row}) {chosen.PointType}");

        Task move = Task.Run(() => host.MoveTo(chosen.coord));
        try
        {
            await move.WaitAsync(TimeSpan.FromSeconds(20));
        }
        catch (TimeoutException)
        {
            Assert.Fail("MoveTo did not return within 20s.");
        }

        _out.WriteLine($"After move: CurrentRoom={rs.CurrentRoom?.GetType().Name} curCoord={rs.CurrentMapCoord} combatInProgress={CombatManager.Instance.IsInProgress}");
    }
}
