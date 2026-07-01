using System;
using System.Linq;
using MegaCrit.Sts2.Core.Map;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises M3 treasure rooms. Reaching the act's treasure node by playing forward depends on
/// combat survival (not yet reliable headless), so these tests jump the run's location straight to
/// the map's real Treasure point — the game's own coord entry doesn't require adjacency — which
/// builds a faithful <see cref="MegaCrit.Sts2.Core.Rooms.TreasureRoom"/> with seeded rewards. The
/// chest flow is then verified through the public option API: opening grants gold, the offered
/// relic surfaces as an explicit take/skip choice, and resolving it returns the player to the map.
/// </summary>
public sealed class TreasureTests
{
    private readonly ITestOutputHelper _out;

    public TreasureTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void TreasureRoom_OpensChest_GrantsGoldAndOffersRelic_ThenTaken()
    {
        GameHost host = EnterTreasureRoom("TESTSEED");

        GameState atTreasure = host.GetState();
        _out.WriteLine($"phase={atTreasure.Phase} floor={atTreasure.Floor} gold={atTreasure.Players[0].Gold} relics=[{string.Join(",", atTreasure.Treasure?.Relics ?? Array.Empty<string>())}]");

        Assert.Equal(GamePhase.Treasure, atTreasure.Phase);
        Assert.NotNull(atTreasure.Treasure);
        Assert.NotEmpty(atTreasure.Treasure!.Relics);

        // Opening the chest granted gold (42-53 before any relic modifiers).
        Assert.True(atTreasure.Players[0].Gold >= 42,
            $"expected chest gold but player has {atTreasure.Players[0].Gold}");

        // The options are exactly one take per offered relic plus a single skip.
        var options = host.ListOptions();
        Assert.Equal(atTreasure.Treasure.Relics.Count, options.Count(o => o.Kind == OptionKind.TakeTreasureRelic));
        Assert.Single(options, o => o.Kind == OptionKind.SkipTreasure);

        int relicsBefore = atTreasure.Players[0].Relics.Count;
        GameOption take = options.First(o => o.Kind == OptionKind.TakeTreasureRelic);
        string relicId = take.TreasureRelicId!;
        _out.WriteLine($"taking treasure relic {relicId}");
        host.Apply(take);

        GameState after = host.GetState();
        _out.WriteLine($"after take: phase={after.Phase} relics=[{string.Join(",", after.Players[0].Relics)}]");

        // The relic was obtained and the player is back on the map (chest resolved).
        Assert.Equal(relicsBefore + 1, after.Players[0].Relics.Count);
        Assert.Contains(relicId, after.Players[0].Relics);
        Assert.Null(after.Treasure);
        Assert.Equal(GamePhase.Map, after.Phase);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);
    }

    [Fact]
    public void TreasureRoom_CanSkip_RelicNotTaken_ReturnsToMap()
    {
        GameHost host = EnterTreasureRoom("TESTSEED");

        GameState atTreasure = host.GetState();
        Assert.Equal(GamePhase.Treasure, atTreasure.Phase);
        int relicsBefore = atTreasure.Players[0].Relics.Count;

        GameOption skip = host.ListOptions().First(o => o.Kind == OptionKind.SkipTreasure);
        host.Apply(skip);

        GameState after = host.GetState();
        _out.WriteLine($"after skip: phase={after.Phase} relics={after.Players[0].Relics.Count} (was {relicsBefore})");

        Assert.Equal(relicsBefore, after.Players[0].Relics.Count);
        Assert.Null(after.Treasure);
        Assert.Equal(GamePhase.Map, after.Phase);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);
    }

    /// <summary>
    /// Start a run, resolve the opening ancient event, then enter the act's treasure node directly.
    /// </summary>
    private static GameHost EnterTreasureRoom(string seed)
    {
        GameHost host = TestNav.StartOnMap(seed);
        MapPointView treasure = host.GetState().Map!.Points
            .First(p => p.PointType == MapPointType.Treasure);
        host.MoveTo(treasure.Coord.ToMapCoord());
        return host;
    }
}
