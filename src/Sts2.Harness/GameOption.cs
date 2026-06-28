using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;

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

    // Live references used by GameHost.Apply. Not part of the serializable surface.
    internal CardModel? CardModel { get; }
    internal Creature? Target { get; }
    internal MapCoord? MapCoord { get; }
    internal Player? Player { get; }

    private GameOption(
        OptionKind kind,
        ulong playerId,
        string description,
        CardView? card = null,
        uint? targetCombatId = null,
        Coord? coord = null,
        CardModel? cardModel = null,
        Creature? target = null,
        MapCoord? mapCoord = null,
        Player? player = null)
    {
        Kind = kind;
        PlayerId = playerId;
        Description = description;
        Card = card;
        TargetCombatId = targetCombatId;
        Coord = coord;
        CardModel = cardModel;
        Target = target;
        MapCoord = mapCoord;
        Player = player;
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
}
