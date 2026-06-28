using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Map;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

public sealed class MapExplorationTests
{
    private readonly ITestOutputHelper _out;

    public MapExplorationTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Map_IsReadable_AndCanMoveIntoFirstRoom()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

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
        bool finished = move.Wait(TimeSpan.FromSeconds(20));
        if (!finished)
        {
            Assert.Fail("MoveTo did not return within 20s.");
        }
        if (move.IsFaulted)
        {
            throw move.Exception!.Flatten().InnerExceptions.First();
        }

        _out.WriteLine($"After move: CurrentRoom={rs.CurrentRoom?.GetType().Name} curCoord={rs.CurrentMapCoord} combatInProgress={CombatManager.Instance.IsInProgress}");
    }
}
