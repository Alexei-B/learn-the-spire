using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rewards;

namespace Sts2.Harness;

/// <summary>
/// The high-level situation the run is in. Determines which kind of options
/// <see cref="GameHost.ListOptions(ulong)"/> returns. Only the phases the harness can
/// currently drive are modelled; rewards/events/shops/rest arrive with later milestones.
/// </summary>
public enum GamePhase
{
    /// <summary>No run is in progress.</summary>
    NotStarted,

    /// <summary>On the map, able to choose the next room to enter.</summary>
    Map,

    /// <summary>In a battle.</summary>
    Combat,

    /// <summary>
    /// The game is blocked on a mid-effect card choice (discover, scry, exhaust, …).
    /// Resolve it via the <see cref="OptionKind.SelectCards"/> options from
    /// <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    Choice,

    /// <summary>
    /// The post-combat rewards screen: gold/potion/relic/card rewards to take, then proceed.
    /// Resolve via the <see cref="OptionKind.TakeReward"/> / <see cref="OptionKind.ProceedFromRewards"/>
    /// options from <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    Reward,

    /// <summary>The run has ended with all players dead.</summary>
    GameOver,

    /// <summary>
    /// A room/screen the harness does not yet model as first-class options
    /// (event, shop, rest, treasure). State is still readable.
    /// </summary>
    Other,
}

/// <summary>
/// An immutable, serializable snapshot of the mechanical game state, projected from the
/// live game logic. This is the read half of the public API: capture it with
/// <see cref="GameHost.GetState"/>, then enumerate moves with
/// <see cref="GameHost.ListOptions(ulong)"/>. Holds no references to live game objects,
/// so it stays valid as a snapshot after the game advances.
/// </summary>
public sealed record GameState
{
    public required GamePhase Phase { get; init; }
    public required string Seed { get; init; }
    public required int ActIndex { get; init; }
    public required int Floor { get; init; }
    public required bool IsGameOver { get; init; }
    public required IReadOnlyList<PlayerState> Players { get; init; }

    /// <summary>The current battle, or null when not in combat.</summary>
    public CombatView? Combat { get; init; }

    /// <summary>The current act's map graph and the player's position on it.</summary>
    public MapView? Map { get; init; }

    /// <summary>A mid-effect card choice the game is blocked on, or null if none.</summary>
    public PendingChoiceView? PendingChoice { get; init; }

    /// <summary>The post-combat rewards on offer, or null when not on the rewards screen.</summary>
    public RewardsView? Rewards { get; init; }
}

/// <summary>The set of post-combat rewards offered to a player.</summary>
public sealed record RewardsView
{
    /// <summary>The rewards in display order; already-taken ones are flagged <see cref="RewardView.Taken"/>.</summary>
    public required IReadOnlyList<RewardView> Rewards { get; init; }
}

/// <summary>A single reward on the rewards screen.</summary>
public sealed record RewardView
{
    public required RewardType Type { get; init; }

    /// <summary>Whether this reward has already been taken.</summary>
    public required bool Taken { get; init; }

    /// <summary>Gold amount for <see cref="RewardType.Gold"/>; null otherwise.</summary>
    public int? Gold { get; init; }

    /// <summary>Potion id for <see cref="RewardType.Potion"/>; null otherwise.</summary>
    public string? PotionId { get; init; }

    /// <summary>Relic id for <see cref="RewardType.Relic"/>; null otherwise.</summary>
    public string? RelicId { get; init; }

    /// <summary>The cards on offer for <see cref="RewardType.Card"/>; null otherwise.</summary>
    public IReadOnlyList<CardView>? Cards { get; init; }
}

/// <summary>A mid-effect card choice the game is waiting on (e.g. discover/exhaust).</summary>
public sealed record PendingChoiceView
{
    /// <summary>The cards to choose from, in the same order as the resolving options.</summary>
    public required IReadOnlyList<CardView> Options { get; init; }

    /// <summary>Minimum cards to select; 0 means the choice may be skipped.</summary>
    public required int MinSelect { get; init; }

    /// <summary>Maximum cards that may be selected.</summary>
    public required int MaxSelect { get; init; }
}

/// <summary>Run-level state for one player (persists across combats).</summary>
public sealed record PlayerState
{
    public required ulong NetId { get; init; }
    public required string Character { get; init; }
    public required int CurrentHp { get; init; }
    public required int MaxHp { get; init; }
    public required int Block { get; init; }
    public required int Gold { get; init; }
    public required int MaxEnergy { get; init; }
    public required IReadOnlyList<CardView> Deck { get; init; }
    public required IReadOnlyList<string> Relics { get; init; }

    /// <summary>One entry per potion slot; null where the slot is empty.</summary>
    public required IReadOnlyList<string?> Potions { get; init; }

    /// <summary>Combat-only state for this player, or null when not in combat.</summary>
    public PlayerCombatView? CombatState { get; init; }
}

/// <summary>Per-player combat state: energy, piles and powers.</summary>
public sealed record PlayerCombatView
{
    public required int Energy { get; init; }
    public required int MaxEnergy { get; init; }
    public required int Stars { get; init; }
    public required int TurnNumber { get; init; }
    public required PlayerTurnPhase Phase { get; init; }
    public required IReadOnlyList<CardView> Hand { get; init; }
    public required IReadOnlyList<CardView> DrawPile { get; init; }
    public required IReadOnlyList<CardView> DiscardPile { get; init; }
    public required IReadOnlyList<CardView> ExhaustPile { get; init; }
    public required IReadOnlyList<PowerView> Powers { get; init; }
}

/// <summary>A single card. <see cref="CardId"/> identifies the model (e.g. "StrikeIronclad").</summary>
public sealed record CardView
{
    public required string CardId { get; init; }
    public required int EnergyCost { get; init; }
    public required bool CostsX { get; init; }
    public required CardType Type { get; init; }
    public required CardRarity Rarity { get; init; }
    public required TargetType TargetType { get; init; }
    public required bool Upgraded { get; init; }

    /// <summary>True only in combat, when the card is currently legal to play.</summary>
    public required bool CanPlay { get; init; }
}

/// <summary>A power (buff/debuff) on a creature, with its current stack amount.</summary>
public sealed record PowerView
{
    public required string PowerId { get; init; }
    public required int Amount { get; init; }
}

/// <summary>The current battle.</summary>
public sealed record CombatView
{
    public required int RoundNumber { get; init; }
    public required CombatSide CurrentSide { get; init; }
    public required IReadOnlyList<EnemyView> Enemies { get; init; }
}

/// <summary>An enemy creature and its telegraphed intent.</summary>
public sealed record EnemyView
{
    public required uint CombatId { get; init; }
    public required string MonsterId { get; init; }
    public required int CurrentHp { get; init; }
    public required int MaxHp { get; init; }
    public required int Block { get; init; }
    public required bool IsHittable { get; init; }
    public required IReadOnlyList<PowerView> Powers { get; init; }
    public required IReadOnlyList<IntentView> Intents { get; init; }
}

/// <summary>One telegraphed action an enemy will take next turn.</summary>
public sealed record IntentView
{
    public required IntentType Type { get; init; }

    /// <summary>Damage per hit for attack intents; null otherwise.</summary>
    public int? Damage { get; init; }

    /// <summary>Number of hits for attack intents; null otherwise.</summary>
    public int? Hits { get; init; }
}

/// <summary>The current act's map graph and the player's position on it.</summary>
public sealed record MapView
{
    public required int ActIndex { get; init; }
    public Coord? CurrentCoord { get; init; }
    public required IReadOnlyList<MapPointView> Points { get; init; }

    /// <summary>The coordinates the player may move to next (room choices).</summary>
    public required IReadOnlyList<Coord> Reachable { get; init; }
}

/// <summary>One point (room) on the map graph.</summary>
public sealed record MapPointView
{
    public required Coord Coord { get; init; }
    public required MapPointType PointType { get; init; }
    public required IReadOnlyList<Coord> Children { get; init; }
}

/// <summary>A map coordinate (column, row).</summary>
public readonly record struct Coord(int Col, int Row)
{
    public static Coord From(MapCoord c) => new(c.col, c.row);
    public MapCoord ToMapCoord() => new(Col, Row);
}
