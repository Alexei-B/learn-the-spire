using System;
using System.Linq;
using MegaCrit.Sts2.Core.Map;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises M3 rest sites. As with treasure rooms, reaching the act's rest node by playing forward
/// isn't reliable yet, so these jump the run location straight to the map's RestSite point (coord
/// entry needs no adjacency), which builds a faithful
/// <see cref="MegaCrit.Sts2.Core.Rooms.RestSiteRoom"/>. The rest actions resolve directly through
/// <c>RestSiteSynchronizer.ChooseLocalOption</c>: Rest heals 30% max HP, and Smith raises a deck
/// card choice (through the same selector seam as combat) that upgrades the picked card.
/// </summary>
public sealed class RestSiteTests
{
    private readonly ITestOutputHelper _out;

    public RestSiteTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void RestSite_Rest_HealsThePlayer_ThenReturnsToMap()
    {
        GameHost host = EnterRestSite("TESTSEED");

        // Hurt the player so a heal has room to act.
        TestNav.SetHp(host, maxHp: 80, currentHp: 40);
        int before = host.GetState().Players[0].CurrentHp;

        GameState atRest = host.GetState();
        _out.WriteLine($"phase={atRest.Phase} options=[{string.Join(",", atRest.RestSite?.Options.Select(o => o.OptionId) ?? Array.Empty<string>())}]");
        Assert.Equal(GamePhase.RestSite, atRest.Phase);
        Assert.NotNull(atRest.RestSite);
        Assert.Contains(atRest.RestSite!.Options, o => o.OptionId == "HEAL");

        GameOption rest = host.ListOptions().First(o => o.RestOptionId == "HEAL");
        host.Apply(rest);

        GameState after = host.GetState();
        _out.WriteLine($"after rest: phase={after.Phase} hp={after.Players[0].CurrentHp} (was {before})");

        // Heal is 30% of max HP (24 here); HP went up and the player is back on the map.
        Assert.True(after.Players[0].CurrentHp > before,
            $"expected HP to rise from {before} but it is {after.Players[0].CurrentHp}");
        Assert.Equal(before + 24, after.Players[0].CurrentHp);
        Assert.Null(after.RestSite);
        Assert.Equal(GamePhase.Map, after.Phase);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);
    }

    [Fact]
    public void RestSite_Smith_UpgradesAChosenCard_ThenReturnsToMap()
    {
        GameHost host = EnterRestSite("TESTSEED");

        GameState atRest = host.GetState();
        Assert.Equal(GamePhase.RestSite, atRest.Phase);
        // The starting deck has upgradable cards, so Smith is offered.
        Assert.Contains(atRest.RestSite!.Options, o => o.OptionId == "SMITH");
        Assert.DoesNotContain(atRest.Players[0].Deck, c => c.Upgraded);

        // Choosing Smith raises a deck card choice (which card to upgrade).
        host.Apply(host.ListOptions().First(o => o.RestOptionId == "SMITH"));
        GameState choosing = host.GetState();
        _out.WriteLine($"phase={choosing.Phase} choiceCards={choosing.PendingChoice?.Options.Count}");
        Assert.Equal(GamePhase.Choice, choosing.Phase);
        Assert.NotNull(choosing.PendingChoice);

        // The forge choice previews every candidate as the upgraded card it would become.
        Assert.True(choosing.PendingChoice!.IsUpgradeSelection);
        Assert.All(choosing.PendingChoice.Options, c => Assert.True(c.Upgraded));
        Assert.All(
            host.ListOptions().Where(o => o.Kind == OptionKind.SelectCards && o.SelectedCards!.Count > 0),
            o => Assert.True(o.Card!.Upgraded));

        // Pick the first upgradable card.
        GameOption pick = host.ListOptions().First(o => o.Kind == OptionKind.SelectCards && o.SelectedCards!.Count > 0);
        string upgradedId = pick.SelectedCards![0].CardId;
        _out.WriteLine($"upgrading {upgradedId}");
        host.Apply(pick);

        GameState after = host.GetState();
        _out.WriteLine($"after smith: phase={after.Phase} upgraded=[{string.Join(",", after.Players[0].Deck.Where(c => c.Upgraded).Select(c => c.CardId))}]");

        // Exactly one card was upgraded and the player is back on the map.
        Assert.Equal(1, after.Players[0].Deck.Count(c => c.Upgraded));
        Assert.Contains(after.Players[0].Deck, c => c.CardId == upgradedId && c.Upgraded);
        Assert.Null(after.RestSite);
        Assert.Equal(GamePhase.Map, after.Phase);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);
    }

    /// <summary>Start a run, resolve the opening ancient event, then enter the act's rest node directly.</summary>
    private static GameHost EnterRestSite(string seed)
    {
        GameHost host = TestNav.StartOnMap(seed);
        MapPointView rest = host.GetState().Map!.Points
            .First(p => p.PointType == MapPointType.RestSite);
        host.MoveTo(rest.Coord.ToMapCoord());
        return host;
    }
}
