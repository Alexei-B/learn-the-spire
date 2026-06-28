using System;
using System.Linq;
using MegaCrit.Sts2.Core.Map;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Exercises M3 merchant shops. As with treasure/rest rooms, reaching the act's shop node by playing
/// forward isn't reliable yet, so these jump the run location straight to the map's Shop point (coord
/// entry needs no adjacency), which builds a faithful
/// <see cref="MegaCrit.Sts2.Core.Rooms.MerchantRoom"/> with a seeded inventory. The player is given
/// gold artificially so every purchase path is reachable. Buying runs the entry's faithful purchase
/// path (pays gold, grants the item); card removal raises a deck card choice through the selector
/// seam; and The Courier relic is used to verify relic hooks change shop behaviour (discounts +
/// slot restock).
/// </summary>
public sealed class ShopTests
{
    private readonly ITestOutputHelper _out;

    public ShopTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Shop_SurfacesInventory_AndAffordableItemsBecomeOptions()
    {
        GameHost host = EnterShop("TESTSEED", gold: 999);

        GameState s = host.GetState();
        _out.WriteLine($"phase={s.Phase} gold={s.Players[0].Gold} items={s.Shop?.Items.Count}");
        Assert.Equal(GamePhase.Shop, s.Phase);
        Assert.NotNull(s.Shop);
        Assert.NotEmpty(s.Shop!.Items);

        // A normal merchant stocks cards, relics, potions and a card-removal service.
        Assert.Contains(s.Shop.Items, i => i.ItemType == "Card");
        Assert.Contains(s.Shop.Items, i => i.ItemType == "Relic");
        Assert.Contains(s.Shop.Items, i => i.ItemType == "Potion");
        // Card removal has a deterministic base price (75, no ascension/jitter).
        ShopItemView removal = Assert.Single(s.Shop.Items, i => i.ItemType == "CardRemoval");
        Assert.Equal(75, removal.Cost);
        foreach (ShopItemView item in s.Shop.Items)
        {
            _out.WriteLine($"  {item.ItemType} {item.ItemId} cost={item.Cost} affordable={item.Affordable}");
        }

        // With plenty of gold every stocked item is affordable and surfaces as a buy option, plus
        // map moves to leave.
        var options = host.ListOptions();
        Assert.Equal(s.Shop.Items.Count, options.Count(o => o.Kind == OptionKind.BuyShopItem));
        Assert.Contains(options, o => o.Kind == OptionKind.MoveTo);
    }

    [Fact]
    public void Shop_BuyCard_AddsToDeck_AndSpendsGold()
    {
        GameHost host = EnterShop("TESTSEED", gold: 999);

        int goldBefore = host.GetState().Players[0].Gold;
        int deckBefore = host.GetState().Players[0].Deck.Count;

        GameOption buyCard = host.ListOptions()
            .First(o => o.Kind == OptionKind.BuyShopItem && o.ShopItemType == "Card");
        string cardId = buyCard.ShopItemId!;
        int cost = buyCard.ShopItemCost!.Value;
        _out.WriteLine($"buying card {cardId} for {cost} (gold {goldBefore})");
        host.Apply(buyCard);

        GameState after = host.GetState();
        Assert.Equal(GamePhase.Shop, after.Phase);
        Assert.Equal(goldBefore - cost, after.Players[0].Gold);
        Assert.Equal(deckBefore + 1, after.Players[0].Deck.Count);
        Assert.Contains(after.Players[0].Deck, c => c.CardId == cardId);
        // The bought card's slot is now empty, so it is no longer offered.
        Assert.DoesNotContain(after.Shop!.Items, i => i.ItemType == "Card" && i.ItemId == cardId);
    }

    [Fact]
    public void Shop_BuyRelic_Obtained_AndSpendsGold()
    {
        GameHost host = EnterShop("TESTSEED", gold: 999);

        int goldBefore = host.GetState().Players[0].Gold;
        int relicsBefore = host.GetState().Players[0].Relics.Count;

        GameOption buyRelic = host.ListOptions()
            .First(o => o.Kind == OptionKind.BuyShopItem && o.ShopItemType == "Relic");
        string relicId = buyRelic.ShopItemId!;
        int cost = buyRelic.ShopItemCost!.Value;
        _out.WriteLine($"buying relic {relicId} for {cost}");
        host.Apply(buyRelic);

        GameState after = host.GetState();
        Assert.Equal(goldBefore - cost, after.Players[0].Gold);
        Assert.Equal(relicsBefore + 1, after.Players[0].Relics.Count);
        Assert.Contains(relicId, after.Players[0].Relics);
    }

    [Fact]
    public void Shop_BuyPotion_Obtained_AndSpendsGold()
    {
        GameHost host = EnterShop("TESTSEED", gold: 999);

        int goldBefore = host.GetState().Players[0].Gold;
        int potionsBefore = host.GetState().Players[0].Potions.Count(p => p is not null);

        GameOption buyPotion = host.ListOptions()
            .First(o => o.Kind == OptionKind.BuyShopItem && o.ShopItemType == "Potion");
        string potionId = buyPotion.ShopItemId!;
        int cost = buyPotion.ShopItemCost!.Value;
        _out.WriteLine($"buying potion {potionId} for {cost}");
        host.Apply(buyPotion);

        GameState after = host.GetState();
        Assert.Equal(goldBefore - cost, after.Players[0].Gold);
        Assert.Equal(potionsBefore + 1, after.Players[0].Potions.Count(p => p is not null));
        Assert.Contains(potionId, after.Players[0].Potions);
    }

    [Fact]
    public void Shop_CardRemoval_RaisesChoice_RemovesCard_SpendsGold_AndIsSingleUse()
    {
        GameHost host = EnterShop("TESTSEED", gold: 999);

        int goldBefore = host.GetState().Players[0].Gold;
        int deckBefore = host.GetState().Players[0].Deck.Count;
        GameOption removal = host.ListOptions()
            .First(o => o.Kind == OptionKind.BuyShopItem && o.ShopItemType == "CardRemoval");
        int cost = removal.ShopItemCost!.Value;
        _out.WriteLine($"buying card removal for {cost}");
        host.Apply(removal);

        // Card removal asks which card to remove (the same selector seam as combat/Smith).
        GameState choosing = host.GetState();
        _out.WriteLine($"phase={choosing.Phase} choiceCards={choosing.PendingChoice?.Options.Count}");
        Assert.Equal(GamePhase.Choice, choosing.Phase);
        Assert.NotNull(choosing.PendingChoice);

        GameOption pick = host.ListOptions()
            .First(o => o.Kind == OptionKind.SelectCards && o.SelectedCards!.Count > 0);
        string removedId = pick.SelectedCards![0].CardId;
        _out.WriteLine($"removing {removedId}");
        host.Apply(pick);

        GameState after = host.GetState();
        Assert.Equal(GamePhase.Shop, after.Phase);
        Assert.Equal(goldBefore - cost, after.Players[0].Gold);
        Assert.Equal(deckBefore - 1, after.Players[0].Deck.Count);
        // The removal service is single-use per shop: it no longer appears, in the view or options.
        Assert.DoesNotContain(after.Shop!.Items, i => i.ItemType == "CardRemoval");
        Assert.DoesNotContain(host.ListOptions(), o => o.ShopItemType == "CardRemoval");
    }

    [Fact]
    public void Shop_Leave_ReturnsToMap()
    {
        GameHost host = EnterShop("TESTSEED", gold: 999);
        Assert.Equal(GamePhase.Shop, host.GetState().Phase);

        GameOption leave = host.ListOptions().First(o => o.Kind == OptionKind.MoveTo);
        host.Apply(leave);

        GameState after = host.GetState();
        _out.WriteLine($"after leaving shop: phase={after.Phase}");
        Assert.Null(after.Shop);
        Assert.NotEqual(GamePhase.Shop, after.Phase);
    }

    [Fact]
    public void Shop_WithTheCourier_DiscountsPrices_AndRestocksAfterPurchase()
    {
        // The Courier (20% discount + refill slots after purchase) is given before entering, so the
        // shop's prices reflect the discount and bought slots restock instead of clearing.
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.GiveRelic(host, "TheCourier");
        TestNav.AddGold(host, 999);
        Assert.Contains("THE_COURIER", host.GetState().Players[0].Relics);

        MapPointView shop = host.GetState().Map!.Points.First(p => p.PointType == MapPointType.Shop);
        host.MoveTo(shop.Coord.ToMapCoord());

        GameState s = host.GetState();
        Assert.Equal(GamePhase.Shop, s.Phase);

        // Discount: the card-removal price has no jitter, so The Courier's flat 20% off is exact:
        // 75 base -> 60. (See Shop_SurfacesInventory, which asserts the undiscounted 75.)
        ShopItemView removal = Assert.Single(s.Shop!.Items, i => i.ItemType == "CardRemoval");
        Assert.Equal(60, removal.Cost);

        int relicSlotsBefore = s.Shop.Items.Count(i => i.ItemType == "Relic");
        Assert.True(relicSlotsBefore > 0);

        // Buy a relic: with The Courier the slot restocks (a fresh relic), so the relic-slot count
        // is unchanged rather than dropping by one.
        GameOption buyRelic = host.ListOptions()
            .First(o => o.Kind == OptionKind.BuyShopItem && o.ShopItemType == "Relic");
        string boughtRelic = buyRelic.ShopItemId!;
        host.Apply(buyRelic);

        GameState after = host.GetState();
        int relicSlotsAfter = after.Shop!.Items.Count(i => i.ItemType == "Relic");
        _out.WriteLine($"relic slots {relicSlotsBefore} -> {relicSlotsAfter}; bought {boughtRelic}; obtained=[{string.Join(",", after.Players[0].Relics)}]");
        Assert.Contains(boughtRelic, after.Players[0].Relics);
        Assert.Equal(relicSlotsBefore, relicSlotsAfter); // restocked, not emptied
    }

    /// <summary>Start a run, resolve the opening ancient event, give gold, then enter the shop node.</summary>
    private static GameHost EnterShop(string seed, int gold)
    {
        GameHost host = TestNav.StartOnMap(seed);
        TestNav.AddGold(host, gold);
        MapPointView shop = host.GetState().Map!.Points.First(p => p.PointType == MapPointType.Shop);
        host.MoveTo(shop.Coord.ToMapCoord());
        return host;
    }
}
