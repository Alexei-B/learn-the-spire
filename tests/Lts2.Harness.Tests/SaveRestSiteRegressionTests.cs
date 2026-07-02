using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Map;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Regression for the save/restore crash on non-combat rooms. A snapshot taken on the map after a
/// rest site (or shop / treasure) used to throw <see cref="ArgumentOutOfRangeException"/> on restore,
/// because the game's <c>AbstractRoom.FromSerializable</c> only reconstructs combat/event rooms.
/// <see cref="GameHost.Snapshot"/> now records no pre-finished room for such rooms, so the save loads
/// and the run stays playable (the last node is re-entered fresh rather than crashing the load).
/// </summary>
public sealed class SaveRestSiteRegressionTests
{
    private readonly ITestOutputHelper _out;

    public SaveRestSiteRegressionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Snapshot_AfterRestSite_RestoresWithoutThrowing()
    {
        await Task.Run(Body).WaitAsync(TimeSpan.FromSeconds(120));
    }

    private void Body()
    {
        // Reach the act's rest node, then rest — returning to the map with the finished RestSiteRoom
        // as the current room (the exact state the broken autosave was captured in).
        GameHost host = TestNav.StartOnMap("RESTSAVE");
        MapPointView rest = host.GetState().Map!.Points.First(p => p.PointType == MapPointType.RestSite);
        host.MoveTo(rest.Coord.ToMapCoord());
        TestNav.SetHp(host, maxHp: 80, currentHp: 40);
        host.Apply(host.ListOptions().First(o => o.RestOptionId == "HEAL"));

        GameState onMap = host.GetState();
        Assert.Equal(GamePhase.Map, onMap.Phase);

        // Round-trip through the JSON save layer the TUI uses. This is what threw before the fix.
        string json = host.ToSaveJson();
        GameHost restored = GameHost.RestoreFromJson(json, "RESTSAVE");

        GameState after = restored.GetState();
        _out.WriteLine($"restored phase={after.Phase} floor={after.Floor} hp={after.Players[0].CurrentHp}");

        // The load succeeded and the run is live and playable.
        Assert.False(after.IsGameOver);
        Assert.NotEmpty(restored.ListOptions());
        Assert.Equal(onMap.Seed, after.Seed);
    }
}
