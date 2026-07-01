using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;

namespace Lts2.Harness;

/// <summary>The kind of action a <see cref="GameOption"/> represents.</summary>
public enum OptionKind
{
    /// <summary>Play a card from hand, optionally at a target enemy.</summary>
    PlayCard,

    /// <summary>End the player's combat turn.</summary>
    EndTurn,

    /// <summary>Move to a reachable map coordinate, entering its room.</summary>
    MoveTo,

    /// <summary>Resolve a pending mid-effect card choice with a (possibly empty) set of cards.</summary>
    SelectCards,

    /// <summary>Choose one option in an event room (e.g. a Neow blessing).</summary>
    ChooseEventOption,

    /// <summary>Take one reward from the post-combat rewards screen (gold/potion/relic/card).</summary>
    TakeReward,

    /// <summary>
    /// Take a card reward's alternative instead of a card: a terminal one (e.g. Pael's Wing's
    /// sacrifice) completes the reward, while a reroll re-rolls the offered cards in place.
    /// </summary>
    TakeCardRewardAlternative,

    /// <summary>Leave the rewards screen, skipping any rewards not taken, and return to the map.</summary>
    ProceedFromRewards,

    /// <summary>Take one of the relics offered in a treasure room.</summary>
    TakeTreasureRelic,

    /// <summary>Skip the treasure room's relics without taking any, then proceed via the map.</summary>
    SkipTreasure,

    /// <summary>Choose a rest-site action (rest/smith/…).</summary>
    ChooseRestOption,

    /// <summary>Buy one item from a merchant shop (card/relic/potion or the card-removal service).</summary>
    BuyShopItem,

    /// <summary>Use (drink/throw) a potion from a belt slot, optionally at a target enemy.</summary>
    UsePotion,

    /// <summary>Discard a potion from a belt slot without using it.</summary>
    DiscardPotion,

    /// <summary>Spend a divination on a Crystal Sphere grid cell (clears it, or a 3×3 area with the Big tool).</summary>
    ClickCrystalSphereCell,

    /// <summary>Switch the Crystal Sphere divination tool (Big = 3×3 area, Small = single cell).</summary>
    SetCrystalSphereTool,
}

/// <summary>
/// A single legal action for the current state, produced by
/// <see cref="GameHost.ListOptions(ulong)"/> and resolved by <see cref="GameHost.Apply"/>.
/// The public fields describe the option; the live game objects it resolves against are
/// held internally. An option is only valid for the state it was listed from — apply it
/// before advancing the game.
/// </summary>
public sealed class GameOption
{
    public OptionKind Kind { get; }
    public ulong PlayerId { get; }

    /// <summary>A short human-readable label, e.g. "Play StrikeIronclad -> Goblin".</summary>
    public string Description { get; }

    /// <summary>The card for <see cref="OptionKind.PlayCard"/>; null otherwise.</summary>
    public CardView? Card { get; }

    /// <summary>The target enemy's combat id for a targeted card play; null otherwise.</summary>
    public uint? TargetCombatId { get; }

    /// <summary>The destination for <see cref="OptionKind.MoveTo"/>; null otherwise.</summary>
    public Coord? Coord { get; }

    /// <summary>The cards this <see cref="OptionKind.SelectCards"/> option will select; empty = skip.</summary>
    public IReadOnlyList<CardView>? SelectedCards { get; }

    /// <summary>The index into the event's options for <see cref="OptionKind.ChooseEventOption"/>; null otherwise.</summary>
    public int? EventOptionIndex { get; }

    /// <summary>The relic an event option grants (e.g. a Neow blessing), if any; null otherwise.</summary>
    public string? EventOptionRelicId { get; }

    /// <summary>The alternative's option id for <see cref="OptionKind.TakeCardRewardAlternative"/> (e.g. "SACRIFICE", "REROLL"); null otherwise.</summary>
    public string? CardRewardAlternativeId { get; }

    /// <summary>The relic id offered by a <see cref="OptionKind.TakeTreasureRelic"/> option; null otherwise.</summary>
    public string? TreasureRelicId { get; }

    /// <summary>The index into the treasure room's relics for <see cref="OptionKind.TakeTreasureRelic"/>; null otherwise.</summary>
    public int? TreasureRelicIndex { get; }

    /// <summary>The rest-site option id for <see cref="OptionKind.ChooseRestOption"/> (e.g. "HEAL"); null otherwise.</summary>
    public string? RestOptionId { get; }

    /// <summary>The index into the rest site's options for <see cref="OptionKind.ChooseRestOption"/>; null otherwise.</summary>
    public int? RestOptionIndex { get; }

    /// <summary>The item type a <see cref="OptionKind.BuyShopItem"/> option buys ("Card"/"Relic"/"Potion"/"CardRemoval"); null otherwise.</summary>
    public string? ShopItemType { get; }

    /// <summary>The model id of the item a <see cref="OptionKind.BuyShopItem"/> option buys; null otherwise.</summary>
    public string? ShopItemId { get; }

    /// <summary>The gold price of a <see cref="OptionKind.BuyShopItem"/> option; null otherwise.</summary>
    public int? ShopItemCost { get; }

    /// <summary>The belt slot index a potion option acts on (<see cref="OptionKind.UsePotion"/>/<see cref="OptionKind.DiscardPotion"/>); null otherwise.</summary>
    public int? PotionSlot { get; }

    /// <summary>The model id of the potion a potion option acts on; null otherwise.</summary>
    public string? PotionId { get; }

    /// <summary>The grid cell a <see cref="OptionKind.ClickCrystalSphereCell"/> option clears; null otherwise.</summary>
    public Coord? CrystalSphereCell { get; }

    /// <summary>The tool a <see cref="OptionKind.SetCrystalSphereTool"/> option selects ("Big"/"Small"); null otherwise.</summary>
    public string? CrystalSphereTool { get; }

    // Live references used by GameHost.Apply. Not part of the serializable surface.
    internal CardModel? CardModel { get; }
    internal Creature? Target { get; }
    internal MapCoord? MapCoord { get; }
    internal Player? Player { get; }
    internal IReadOnlyList<CardModel>? SelectedCardModels { get; }

    /// <summary>The tool a <see cref="OptionKind.SetCrystalSphereTool"/> option selects; null otherwise.</summary>
    internal MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType? CrystalSphereToolValue { get; }

    /// <summary>The reward this <see cref="OptionKind.TakeReward"/> option claims; null otherwise.</summary>
    internal Reward? Reward { get; }

    /// <summary>The live alternative a <see cref="OptionKind.TakeCardRewardAlternative"/> option runs; null otherwise.</summary>
    internal MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative? CardRewardAlternativeModel { get; }

    /// <summary>The shop entry a <see cref="OptionKind.BuyShopItem"/> option purchases; null otherwise.</summary>
    internal MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry? ShopEntry { get; }

    /// <summary>The potion a potion option acts on; null otherwise.</summary>
    internal PotionModel? PotionModel { get; }

    private GameOption(
        OptionKind kind,
        ulong playerId,
        string description,
        CardView? card = null,
        uint? targetCombatId = null,
        Coord? coord = null,
        IReadOnlyList<CardView>? selectedCards = null,
        int? eventOptionIndex = null,
        string? eventOptionRelicId = null,
        string? cardRewardAlternativeId = null,
        string? treasureRelicId = null,
        int? treasureRelicIndex = null,
        string? restOptionId = null,
        int? restOptionIndex = null,
        string? shopItemType = null,
        string? shopItemId = null,
        int? shopItemCost = null,
        int? potionSlot = null,
        string? potionId = null,
        Coord? crystalSphereCell = null,
        string? crystalSphereTool = null,
        CardModel? cardModel = null,
        Creature? target = null,
        MapCoord? mapCoord = null,
        Player? player = null,
        IReadOnlyList<CardModel>? selectedCardModels = null,
        Reward? reward = null,
        MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative? cardRewardAlternativeModel = null,
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry? shopEntry = null,
        PotionModel? potionModel = null,
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType? crystalSphereToolValue = null)
    {
        Kind = kind;
        PlayerId = playerId;
        Description = description;
        Card = card;
        TargetCombatId = targetCombatId;
        Coord = coord;
        SelectedCards = selectedCards;
        EventOptionIndex = eventOptionIndex;
        EventOptionRelicId = eventOptionRelicId;
        CardRewardAlternativeId = cardRewardAlternativeId;
        TreasureRelicId = treasureRelicId;
        TreasureRelicIndex = treasureRelicIndex;
        RestOptionId = restOptionId;
        RestOptionIndex = restOptionIndex;
        ShopItemType = shopItemType;
        ShopItemId = shopItemId;
        ShopItemCost = shopItemCost;
        PotionSlot = potionSlot;
        PotionId = potionId;
        CrystalSphereCell = crystalSphereCell;
        CrystalSphereTool = crystalSphereTool;
        CardModel = cardModel;
        Target = target;
        MapCoord = mapCoord;
        Player = player;
        SelectedCardModels = selectedCardModels;
        Reward = reward;
        CardRewardAlternativeModel = cardRewardAlternativeModel;
        ShopEntry = shopEntry;
        PotionModel = potionModel;
        CrystalSphereToolValue = crystalSphereToolValue;
    }

    internal static GameOption PlayCardOption(Player player, CardModel cardModel, CardView card, Creature? target)
    {
        string desc = target?.Monster is not null
            ? $"Play {card.CardId} -> {target.Monster.Id.Entry}"
            : $"Play {card.CardId}";
        return new GameOption(
            OptionKind.PlayCard, player.NetId, desc,
            card: card, targetCombatId: target?.CombatId,
            cardModel: cardModel, target: target, player: player);
    }

    internal static GameOption EndTurnOption(Player player) =>
        new(OptionKind.EndTurn, player.NetId, "End turn", player: player);

    internal static GameOption MoveToOption(Player player, MapCoord coord) =>
        new(OptionKind.MoveTo, player.NetId, $"Move to ({coord.col},{coord.row})",
            coord: Lts2.Harness.Coord.From(coord), mapCoord: coord, player: player);

    internal static GameOption SelectCardsOption(
        Player player, IReadOnlyList<CardModel> cardModels, IReadOnlyList<CardView> cards)
    {
        string desc = cardModels.Count == 0
            ? "Skip selection"
            : "Select " + string.Join(", ", cardModels.Select(c => c.Id.Entry));
        return new GameOption(
            OptionKind.SelectCards, player.NetId, desc,
            selectedCards: cards, selectedCardModels: cardModels, player: player);
    }

    /// <summary>
    /// Choose an event option at the given index into the event's option list. Carries the
    /// option's loc key and any relic it grants for display.
    /// </summary>
    internal static GameOption ChooseEventOption(
        Player player, int index, MegaCrit.Sts2.Core.Events.EventOption option)
    {
        string? relicId = option.Relic?.Id.Entry;
        string desc = relicId is not null
            ? $"Event option {option.TextKey} (relic {relicId})"
            : $"Event option {option.TextKey}";
        return new GameOption(
            OptionKind.ChooseEventOption, player.NetId, desc,
            eventOptionIndex: index, eventOptionRelicId: relicId, player: player);
    }

    /// <summary>
    /// Take a non-card reward (gold/potion/relic) from the rewards screen.
    /// </summary>
    internal static GameOption TakeRewardOption(Player player, Reward reward, string description) =>
        new(OptionKind.TakeReward, player.NetId, description, player: player, reward: reward);

    /// <summary>
    /// Take a card reward by picking one of its offered cards (added to the deck).
    /// </summary>
    internal static GameOption TakeCardRewardOption(Player player, Reward reward, CardModel cardToPick, CardView card) =>
        new(OptionKind.TakeReward, player.NetId, $"Take card {cardToPick.Id.Entry}",
            card: card, cardModel: cardToPick, player: player, reward: reward);

    /// <summary>
    /// Take a card reward's alternative (e.g. Pael's Wing's "SACRIFICE", or "REROLL") instead of a
    /// card. Carries the live reward and alternative so <see cref="GameHost.Apply"/> can run it.
    /// </summary>
    internal static GameOption TakeCardRewardAlternativeOption(
        Player player, Reward reward, MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative alternative) =>
        new(OptionKind.TakeCardRewardAlternative, player.NetId, $"Card reward: {alternative.OptionId}",
            cardRewardAlternativeId: alternative.OptionId, player: player,
            reward: reward, cardRewardAlternativeModel: alternative);

    internal static GameOption ProceedFromRewardsOption(Player player) =>
        new(OptionKind.ProceedFromRewards, player.NetId, "Proceed (leave rewards)", player: player);

    /// <summary>Take the relic at the given index in the treasure room's relic offering.</summary>
    internal static GameOption TakeTreasureRelicOption(Player player, int index, string relicId) =>
        new(OptionKind.TakeTreasureRelic, player.NetId, $"Take treasure relic {relicId}",
            treasureRelicId: relicId, treasureRelicIndex: index, player: player);

    internal static GameOption SkipTreasureOption(Player player) =>
        new(OptionKind.SkipTreasure, player.NetId, "Skip treasure relics", player: player);

    /// <summary>Choose the rest-site option at the given index (e.g. rest or smith).</summary>
    internal static GameOption ChooseRestOption(Player player, int index, string optionId) =>
        new(OptionKind.ChooseRestOption, player.NetId, $"Rest option {optionId}",
            restOptionId: optionId, restOptionIndex: index, player: player);

    /// <summary>
    /// Buy one item from a merchant shop. Carries the item's type/id/cost for display and the live
    /// <see cref="MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry"/> the purchase resolves against.
    /// </summary>
    internal static GameOption BuyShopItemOption(
        Player player, MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry,
        string itemType, string itemId, CardView? card)
    {
        int cost = entry.Cost;
        string desc = $"Buy {itemType} {itemId} ({cost} gold)";
        return new GameOption(
            OptionKind.BuyShopItem, player.NetId, desc,
            card: card, shopItemType: itemType, shopItemId: itemId, shopItemCost: cost,
            player: player, shopEntry: entry);
    }

    /// <summary>
    /// Use (drink/throw) the potion in the given belt slot, optionally at a target enemy. Carries the
    /// live potion for resolution and the target's combat id for display.
    /// </summary>
    internal static GameOption UsePotionOption(Player player, int slot, PotionModel potion, Creature? target)
    {
        string desc = target?.Monster is not null
            ? $"Use potion {potion.Id.Entry} -> {target.Monster.Id.Entry}"
            : $"Use potion {potion.Id.Entry}";
        return new GameOption(
            OptionKind.UsePotion, player.NetId, desc,
            targetCombatId: target?.CombatId, potionSlot: slot, potionId: potion.Id.Entry,
            target: target, player: player, potionModel: potion);
    }

    /// <summary>Discard the potion in the given belt slot without using it.</summary>
    internal static GameOption DiscardPotionOption(Player player, int slot, PotionModel potion) =>
        new(OptionKind.DiscardPotion, player.NetId, $"Discard potion {potion.Id.Entry}",
            potionSlot: slot, potionId: potion.Id.Entry, player: player, potionModel: potion);

    /// <summary>Spend a divination clearing the Crystal Sphere grid cell at (x, y) with the active tool.</summary>
    internal static GameOption ClickCrystalSphereCellOption(Player player, int x, int y) =>
        new(OptionKind.ClickCrystalSphereCell, player.NetId, $"Divine cell ({x},{y})",
            crystalSphereCell: new Coord(x, y), player: player);

    /// <summary>Switch the Crystal Sphere divination tool.</summary>
    internal static GameOption SetCrystalSphereToolOption(
        Player player, MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType tool) =>
        new(OptionKind.SetCrystalSphereTool, player.NetId, $"Use {tool} divination tool",
            crystalSphereTool: tool.ToString(), player: player, crystalSphereToolValue: tool);
}
