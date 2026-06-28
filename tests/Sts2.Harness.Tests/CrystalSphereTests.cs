using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Exercises the Crystal Sphere event minigame end-to-end through the public option API. The game
/// drives this minigame through a UI screen (null headless); the harness skips that screen and
/// surfaces the grid as <see cref="GamePhase.CrystalSphere"/> with one
/// <see cref="OptionKind.ClickCrystalSphereCell"/> per hidden cell plus a tool toggle. These tests
/// verify the core rule the player sees: an item only pays out when its whole footprint is uncovered
/// — partially uncovering it does not count.
/// </summary>
public sealed class CrystalSphereTests
{
    private readonly ITestOutputHelper _out;

    public CrystalSphereTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void FullyUncoveringAnItem_RevealsIt_PartialDoesNot()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        EnterMinigameViaPaymentPlan(host);

        // Switch to the Small (single-cell) tool so we can uncover exactly the cells we choose.
        host.Apply(host.ListOptions().Single(o =>
            o.Kind == OptionKind.SetCrystalSphereTool && o.CrystalSphereTool == "Small"));

        CrystalSphereView view = host.GetState().CrystalSphere!;
        _out.WriteLine($"grid {view.Width}x{view.Height} divinations={view.DivinationsLeft} tool={view.Tool} items={view.Items.Count}");

        // Pick the smallest multi-cell item: small enough to fit the divination budget, but large
        // enough that leaving one cell covered is a genuine "partial" state to test against.
        CrystalSphereItemView target = view.Items
            .Where(i => i.Size.Col * i.Size.Row >= 2)
            .OrderBy(i => i.Size.Col * i.Size.Row)
            .First();
        _out.WriteLine($"target {target.ItemType} at ({target.Position.Col},{target.Position.Row}) size {target.Size.Col}x{target.Size.Row}");
        Assert.False(target.Revealed, "items start fully hidden");

        List<Coord> footprint = FootprintCells(target).ToList();
        Assert.True(footprint.Count <= view.DivinationsLeft, "test needs the item to fit in the divination budget");

        // Uncover all but the last cell: the item is still partially covered, so it must NOT count.
        for (int i = 0; i < footprint.Count - 1; i++)
        {
            ClickCell(host, footprint[i]);
        }
        Assert.True(host.GetState().Phase == GamePhase.CrystalSphere, "minigame should still be in progress");
        Assert.False(
            CurrentItem(host, target.Position).Revealed,
            "an item with one cell still under fog must not be revealed");

        // Uncover the final cell: now the whole footprint is clear, so the item is revealed.
        ClickCell(host, footprint[^1]);
        Assert.True(
            CurrentItem(host, target.Position).Revealed,
            "an item with every cell uncovered must be revealed");
    }

    [Fact]
    public void PlayingTheMinigameOut_GrantsRewardsForRevealedItems_AndReturnsToTheMap()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        int relicsBefore = host.GetState().Players[0].Relics.Count;
        int deckBefore = host.GetState().Players[0].Deck.Count;
        EnterMinigameViaPaymentPlan(host);

        // Greedily spend every divination uncovering items, then take the rewards the revealed items
        // pay out (which surface through the normal custom-reward screen) until back on the map.
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
            maxSteps: 200,
            log: _out);

        PlayerState player = end.Players[0];
        _out.WriteLine($"ended phase={end.Phase} gold={player.Gold} relics={player.Relics.Count} deck={player.Deck.Count}");

        Assert.Equal(GamePhase.Map, end.Phase);
        Assert.Null(end.CrystalSphere);
        // PaymentPlan adds a curse to the deck, so the deck must have grown (plus any card the greedy
        // play uncovered and took) — evidence the minigame resolved and paid out through to the map.
        Assert.True(player.Deck.Count > deckBefore,
            $"expected the deck to grow (curse + any revealed card) but it went {deckBefore} -> {player.Deck.Count}");
        Assert.True(player.Relics.Count >= relicsBefore, "relic count should not regress through the minigame");
    }

    /// <summary>
    /// Resolve the run to the Crystal Sphere event's PaymentPlan option, which (unlike UncoverFuture)
    /// needs no gold — it adds a Debt curse — and then opens the minigame.
    /// </summary>
    private void EnterMinigameViaPaymentPlan(GameHost host)
    {
        EventModel ev = Act1EventsTests.ResolveEvent("CrystalSphere");
        host.EnterEventDebug(ev);

        GameState atEvent = host.GetState();
        Assert.Equal(GamePhase.Event, atEvent.Phase);
        int paymentPlanIndex = atEvent.Event!.Options
            .Single(o => o.TextKey.Contains("PAYMENT_PLAN")).Index;

        host.Apply(host.ListOptions().Single(o =>
            o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == paymentPlanIndex));

        Assert.Equal(GamePhase.CrystalSphere, host.GetState().Phase);
    }

    private static IEnumerable<Coord> FootprintCells(CrystalSphereItemView item)
    {
        for (int i = 0; i < item.Size.Col; i++)
        {
            for (int j = 0; j < item.Size.Row; j++)
            {
                yield return new Coord(item.Position.Col + i, item.Position.Row + j);
            }
        }
    }

    private static CrystalSphereItemView CurrentItem(GameHost host, Coord position) =>
        host.GetState().CrystalSphere!.Items.Single(i => i.Position == position);

    private static void ClickCell(GameHost host, Coord cell) =>
        host.Apply(host.ListOptions().Single(o =>
            o.Kind == OptionKind.ClickCrystalSphereCell && o.CrystalSphereCell == cell));
}
