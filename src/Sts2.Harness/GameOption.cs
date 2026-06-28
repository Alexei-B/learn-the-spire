using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;

namespace Sts2.Harness;

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

    /// <summary>Leave the rewards screen, skipping any rewards not taken, and return to the map.</summary>
    ProceedFromRewards,

    /// <summary>Take one of the relics offered in a treasure room.</summary>
    TakeTreasureRelic,

    /// <summary>Skip the treasure room's relics without taking any, then proceed via the map.</summary>
    SkipTreasure,
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

    /// <summary>The relic id offered by a <see cref="OptionKind.TakeTreasureRelic"/> option; null otherwise.</summary>
    public string? TreasureRelicId { get; }

    /// <summary>The index into the treasure room's relics for <see cref="OptionKind.TakeTreasureRelic"/>; null otherwise.</summary>
    public int? TreasureRelicIndex { get; }

    // Live references used by GameHost.Apply. Not part of the serializable surface.
    internal CardModel? CardModel { get; }
    internal Creature? Target { get; }
    internal MapCoord? MapCoord { get; }
    internal Player? Player { get; }
    internal IReadOnlyList<CardModel>? SelectedCardModels { get; }

    /// <summary>The reward this <see cref="OptionKind.TakeReward"/> option claims; null otherwise.</summary>
    internal Reward? Reward { get; }

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
        string? treasureRelicId = null,
        int? treasureRelicIndex = null,
        CardModel? cardModel = null,
        Creature? target = null,
        MapCoord? mapCoord = null,
        Player? player = null,
        IReadOnlyList<CardModel>? selectedCardModels = null,
        Reward? reward = null)
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
        TreasureRelicId = treasureRelicId;
        TreasureRelicIndex = treasureRelicIndex;
        CardModel = cardModel;
        Target = target;
        MapCoord = mapCoord;
        Player = player;
        SelectedCardModels = selectedCardModels;
        Reward = reward;
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
            coord: Sts2.Harness.Coord.From(coord), mapCoord: coord, player: player);

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

    internal static GameOption ProceedFromRewardsOption(Player player) =>
        new(OptionKind.ProceedFromRewards, player.NetId, "Proceed (leave rewards)", player: player);

    /// <summary>Take the relic at the given index in the treasure room's relic offering.</summary>
    internal static GameOption TakeTreasureRelicOption(Player player, int index, string relicId) =>
        new(OptionKind.TakeTreasureRelic, player.NetId, $"Take treasure relic {relicId}",
            treasureRelicId: relicId, treasureRelicIndex: index, player: player);

    internal static GameOption SkipTreasureOption(Player player) =>
        new(OptionKind.SkipTreasure, player.NetId, "Skip treasure relics", player: player);
}
