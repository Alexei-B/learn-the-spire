using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rewards;

namespace Lts2.Harness;

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

    /// <summary>
    /// In an event room awaiting a choice (e.g. the run-opening Neow ancient event). Resolve via
    /// the <see cref="OptionKind.ChooseEventOption"/> options from <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    Event,

    /// <summary>
    /// Blocked on a "choose a bundle" selection (ScrollBoxes: pick one of the offered card bundles for
    /// your deck). Resolve via the <see cref="OptionKind.ChooseBundle"/> options from
    /// <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    BundleChoice,

    /// <summary>
    /// In a treasure room with the chest opened and relics still to pick. Resolve via the
    /// <see cref="OptionKind.TakeTreasureRelic"/> / <see cref="OptionKind.SkipTreasure"/> options
    /// from <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    Treasure,

    /// <summary>
    /// At a rest site with a rest action still available (rest/smith/…). Resolve via the
    /// <see cref="OptionKind.ChooseRestOption"/> options from <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    RestSite,

    /// <summary>
    /// In a merchant shop: cards/relics/potions to buy, a card-removal service, and the freedom
    /// to leave by moving on. Resolve via the <see cref="OptionKind.BuyShopItem"/> options (and a
    /// <see cref="OptionKind.MoveTo"/> to leave) from <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    Shop,

    /// <summary>
    /// Playing the Crystal Sphere event minigame: spend divinations to uncover grid cells, fully
    /// revealing items to earn their rewards. Resolve via the
    /// <see cref="OptionKind.ClickCrystalSphereCell"/> / <see cref="OptionKind.SetCrystalSphereTool"/>
    /// options from <see cref="GameHost.ListOptions(ulong)"/>.
    /// </summary>
    CrystalSphere,

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

    /// <summary>
    /// The run's ascension level (0–10). Higher levels add difficulty modifiers — swarming elites,
    /// a weaker Neow heal, poverty (less gold), a tighter potion belt, the Ascender's Bane curse,
    /// inflation, scarcity, tougher/deadlier enemies and a double final boss — applied by the game's
    /// <c>AscensionManager</c>. Each level is cumulative (a level-N run has every modifier up to N).
    /// </summary>
    public required int AscensionLevel { get; init; }

    public required bool IsGameOver { get; init; }

    /// <summary>
    /// True once the run has been won (the act-3 boss is beaten and the Architect's victory event
    /// has ended the run). A won run is also <see cref="IsGameOver"/> — the game ends a victory by
    /// killing the players — so this distinguishes a victorious end from a death.
    /// </summary>
    public required bool IsVictory { get; init; }

    /// <summary>
    /// The run's score so far (<c>ScoreUtility.CalculateScore</c>): floors climbed, gold gained,
    /// elites and bosses slain, scaled by ascension. Computed with the win flag set once the run is
    /// a victory, so on a finished run it is the final score (a bit higher on a win — the act-3 boss
    /// only counts as slain once the run is won).
    /// </summary>
    public required int Score { get; init; }
    public required IReadOnlyList<PlayerState> Players { get; init; }

    /// <summary>The current battle, or null when not in combat.</summary>
    public CombatView? Combat { get; init; }

    /// <summary>The current act's map graph and the player's position on it.</summary>
    public MapView? Map { get; init; }

    /// <summary>A mid-effect card choice the game is blocked on, or null if none.</summary>
    public PendingChoiceView? PendingChoice { get; init; }

    /// <summary>The "choose a bundle" selection (ScrollBoxes) on offer, or null when there is none.</summary>
    public BundleChoiceView? BundleChoice { get; init; }

    /// <summary>The post-combat rewards on offer, or null when not on the rewards screen.</summary>
    public RewardsView? Rewards { get; init; }

    /// <summary>The event in progress and its choices, or null when not in an actionable event.</summary>
    public EventView? Event { get; init; }

    /// <summary>The treasure-room relics to pick from, or null when not in a treasure room.</summary>
    public TreasureView? Treasure { get; init; }

    /// <summary>The rest-site options available, or null when not at a rest site.</summary>
    public RestSiteView? RestSite { get; init; }

    /// <summary>The merchant shop inventory, or null when not in a shop.</summary>
    public ShopView? Shop { get; init; }

    /// <summary>The Crystal Sphere minigame in progress, or null when not playing it.</summary>
    public CrystalSphereView? CrystalSphere { get; init; }
}

/// <summary>
/// The Crystal Sphere event minigame: a fogged grid with hidden items. Each divination clears a
/// cell (or a 3×3 area with the Big tool); an item whose every cell is uncovered is revealed and
/// pays out (as a custom reward set) once divinations run out.
/// </summary>
public sealed record CrystalSphereView
{
    /// <summary>Grid width (columns).</summary>
    public required int Width { get; init; }

    /// <summary>Grid height (rows).</summary>
    public required int Height { get; init; }

    /// <summary>Divinations (cell-clears) the player still has.</summary>
    public required int DivinationsLeft { get; init; }

    /// <summary>The active tool: "Big" clears a 3×3 area, "Small" clears a single cell.</summary>
    public required string Tool { get; init; }

    /// <summary>The cells still hidden under fog (the clickable cells).</summary>
    public required IReadOnlyList<Coord> HiddenCells { get; init; }

    /// <summary>The items hidden on the grid, with their footprint and whether fully uncovered.</summary>
    public required IReadOnlyList<CrystalSphereItemView> Items { get; init; }
}

/// <summary>One item hidden on the Crystal Sphere grid.</summary>
public sealed record CrystalSphereItemView
{
    /// <summary>The item kind (e.g. "CardReward", "Relic", "Potion", "Gold", "Curse").</summary>
    public required string ItemType { get; init; }

    /// <summary>True for beneficial items; false for the curse.</summary>
    public required bool IsGood { get; init; }

    /// <summary>The top-left cell of the item's footprint.</summary>
    public required Coord Position { get; init; }

    /// <summary>The item's footprint size (columns, rows).</summary>
    public required Coord Size { get; init; }

    /// <summary>True once every cell of the footprint is uncovered (the reward is earned).</summary>
    public required bool Revealed { get; init; }
}

/// <summary>
/// A merchant shop's inventory: the cards, relics and potions on offer, plus the card-removal
/// service. Each item carries its price and whether the player can currently afford it. Only
/// affordable, in-stock items surface as <see cref="OptionKind.BuyShopItem"/> options; this view
/// lists the full inventory so an agent can see what is unaffordable too.
/// </summary>
public sealed record ShopView
{
    /// <summary>The player's current gold (mirrors <see cref="PlayerState.Gold"/> for convenience).</summary>
    public required int Gold { get; init; }

    /// <summary>Every still-stocked item on offer (cards/relics/potions + card removal).</summary>
    public required IReadOnlyList<ShopItemView> Items { get; init; }
}

/// <summary>One purchasable entry in a shop.</summary>
public sealed record ShopItemView
{
    /// <summary>The kind of item: "Card", "Relic", "Potion", or "CardRemoval".</summary>
    public required string ItemType { get; init; }

    /// <summary>The model id of the card/relic/potion, or "CardRemoval" for the removal service.</summary>
    public required string ItemId { get; init; }

    /// <summary>The current price in gold (after relic discounts like The Courier).</summary>
    public required int Cost { get; init; }

    /// <summary>True when the player has enough gold to buy it.</summary>
    public required bool Affordable { get; init; }

    /// <summary>The card on offer, for an item of type "Card"; null otherwise.</summary>
    public CardView? Card { get; init; }
}

/// <summary>A rest site with the rest actions still available to choose.</summary>
public sealed record RestSiteView
{
    /// <summary>The selectable options (rest/smith/…), in the same order as the resolving options.</summary>
    public required IReadOnlyList<RestSiteOptionView> Options { get; init; }
}

/// <summary>One selectable rest-site action.</summary>
public sealed record RestSiteOptionView
{
    /// <summary>The index passed back to resolve it (its position in the live option list).</summary>
    public required int Index { get; init; }

    /// <summary>The option id (e.g. "HEAL", "SMITH").</summary>
    public required string OptionId { get; init; }
}

/// <summary>A treasure room with its chest opened: the relics still available to pick.</summary>
public sealed record TreasureView
{
    /// <summary>
    /// The relics on offer, in index order (the index is what <see cref="OptionKind.TakeTreasureRelic"/>
    /// resolves against). Singleplayer offers exactly one; a multi-player chest offers one per player,
    /// and the players vote — each picks one (or skips), conflicts resolved by the game.
    /// </summary>
    public required IReadOnlyList<string> Relics { get; init; }

    /// <summary>
    /// In a multi-player chest, what each player is currently indicating — their pending relic vote —
    /// so an agent can see the others' picks before the chest resolves. Empty in single-player.
    /// </summary>
    public required IReadOnlyList<TreasureVoteView> Votes { get; init; }
}

/// <summary>One player's pending pick at a multi-player treasure chest.</summary>
public sealed record TreasureVoteView
{
    public required ulong NetId { get; init; }

    /// <summary>True once this player has cast their pick (a relic or a skip).</summary>
    public required bool HasVoted { get; init; }

    /// <summary>The relic index this player picked, or null if they skipped or have not picked yet.</summary>
    public int? VotedRelicIndex { get; init; }
}

/// <summary>An event room awaiting a choice, with its currently-offered options.</summary>
public sealed record EventView
{
    /// <summary>The event model id (e.g. "NEOW").</summary>
    public required string EventId { get; init; }

    /// <summary>
    /// The event's current body/description text (raw game markup, with dynamic numbers filled in),
    /// or null when the event has none. This is the live current-page description — the flavour text
    /// shown above the options — rendered from the running event so its numbers are correct.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>True when this is an ancient event (Neow-style run-start / map node).</summary>
    public required bool IsAncient { get; init; }

    /// <summary>
    /// True when this is a shared (vote-based) event: all players vote on a single option and the game
    /// picks one once everyone has voted. False for a per-player event, where each player resolves
    /// their own instance independently.
    /// </summary>
    public required bool IsShared { get; init; }

    /// <summary>The selectable options, in the same order as the resolving options.</summary>
    public required IReadOnlyList<EventOptionView> Options { get; init; }

    /// <summary>
    /// For a shared event, what each player is currently indicating — their pending vote — so an agent
    /// can see the others' choices before the vote resolves. Empty for a per-player event.
    /// </summary>
    public required IReadOnlyList<EventVoteView> Votes { get; init; }
}

/// <summary>One player's pending vote on a shared event.</summary>
public sealed record EventVoteView
{
    public required ulong NetId { get; init; }

    /// <summary>True once this player has cast a vote this round.</summary>
    public required bool HasVoted { get; init; }

    /// <summary>The option index this player voted for, or null if they have not voted yet.</summary>
    public int? VotedOptionIndex { get; init; }
}

/// <summary>One selectable option on an event screen.</summary>
public sealed record EventOptionView
{
    /// <summary>The index into the event's option list (the value passed back to resolve it).</summary>
    public required int Index { get; init; }

    /// <summary>The option's localization key (stable identifier for the choice).</summary>
    public required string TextKey { get; init; }

    /// <summary>
    /// The option's live title text (raw game markup with bound dynamic numbers), or null when it has
    /// none. Rendered from the running event option, so any per-run numbers are correct. Prefer this
    /// over a by-key re-lookup, which misses the live dynamic-variable bindings.
    /// </summary>
    public string? Title { get; init; }

    /// <summary>
    /// The option's live outcome/description text (raw game markup with bound dynamic numbers), or
    /// null when it has none (e.g. "Choose an Attack to Enchant with Sharp 2"). Rendered from the
    /// running event option so its numbers reflect this run.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>The relic this option grants, if any; null otherwise.</summary>
    public string? RelicId { get; init; }
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

    /// <summary>
    /// For a <see cref="RewardType.Card"/> reward, the option ids of its non-skip alternatives
    /// (e.g. "SACRIFICE", "REROLL"), each resolvable via an
    /// <see cref="OptionKind.TakeCardRewardAlternative"/> option; null/empty otherwise.
    /// </summary>
    public IReadOnlyList<string>? CardAlternatives { get; init; }
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

    /// <summary>True when this choice picks a card to upgrade (the rest-site forge); its options are
    /// shown as the upgraded card they would become.</summary>
    public bool IsUpgradeSelection { get; init; }
}

/// <summary>
/// A "choose a bundle" selection (ScrollBoxes): pick one of several card bundles, whose cards are all
/// added to the deck. Resolve via the <see cref="OptionKind.ChooseBundle"/> options.
/// </summary>
public sealed record BundleChoiceView
{
    /// <summary>The bundles on offer, in the same order as the resolving options; each is a group of
    /// cards taken together.</summary>
    public required IReadOnlyList<IReadOnlyList<CardView>> Bundles { get; init; }
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

    /// <summary>
    /// The Defect's channeled orbs, oldest first (the Regent/others gain these only via cross-character
    /// cards). Empty when the player has no orb slots. <see cref="OrbSlots"/> is the total capacity, so
    /// the UI can draw the empty slots too.
    /// </summary>
    public IReadOnlyList<OrbView> Orbs { get; init; } = System.Array.Empty<OrbView>();
    public int OrbSlots { get; init; }

    /// <summary>
    /// The Necrobinder's Osty pet, when it is present in this combat (null for other characters, or
    /// before Osty is summoned). Stays non-null but <see cref="OstyView.IsAlive"/>=false once Osty
    /// has died this combat.
    /// </summary>
    public OstyView? Osty { get; init; }
}

/// <summary>One channeled orb in the Defect's orb queue.</summary>
public sealed record OrbView
{
    public required string OrbId { get; init; }

    /// <summary>The orb's passive (per-turn) value — e.g. Lightning damage, Frost block, Plasma energy.</summary>
    public required int PassiveValue { get; init; }

    /// <summary>The orb's evoke value — for a Dark orb this is its accumulated charge.</summary>
    public required int EvokeValue { get; init; }
}

/// <summary>The Necrobinder's Osty pet creature (a combat-only ally with its own HP/block).</summary>
public sealed record OstyView
{
    public required int CurrentHp { get; init; }
    public required int MaxHp { get; init; }
    public required int Block { get; init; }
    public required bool IsAlive { get; init; }
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

    /// <summary>
    /// In combat, the card's actual per-hit attack damage after all modifiers (self powers like
    /// Strength/Weak/Vigor and, when projected against a specific target, that target's
    /// Vulnerable/Intangible/etc.), with <see cref="BaseDamage"/> the unmodified printed value.
    /// Null for non-attacks or outside combat. Compare the two to colour it (more = buffed, less = weakened).
    /// </summary>
    public int? Damage { get; init; }
    public int? BaseDamage { get; init; }

    /// <summary>
    /// In combat, the card's actual Block after modifiers (Frail/Dexterity/…), with
    /// <see cref="BaseBlock"/> the unmodified printed value. Null for cards that grant no block.
    /// </summary>
    public int? Block { get; init; }
    public int? BaseBlock { get; init; }

    /// <summary>
    /// In combat, the block-equivalent value a card grants beyond its printed <see cref="Block"/> — a
    /// character-specific defensive mechanic the default strategy should weigh like block. Currently the
    /// Necrobinder's Osty summon amount (Osty is a wall that soaks hits). Null when the card summons
    /// nothing. (Defect orb block and Silent sly are not modelled here yet.)
    /// </summary>
    public int? Summon { get; init; }

    /// <summary>The card's star cost (the Regent's second resource), or 0/negative for cards that
    /// don't cost stars. The UI shows a ★ badge only when this is positive.</summary>
    public int StarCost { get; init; }

    /// <summary>The id of an enchantment applied to this card (e.g. Corrupted), or null.</summary>
    public string? EnchantmentId { get; init; }

    /// <summary>The id of an affliction applied to this card (e.g. Bound), or null.</summary>
    public string? AfflictionId { get; init; }

    /// <summary>How many extra times this card replays when played (0 = none), from an enchantment or
    /// a granted effect.</summary>
    public int ReplayCount { get; init; }

    /// <summary>Keywords granted to this card by some effect beyond its printed ones (e.g. Retain from
    /// Transfigure, Ethereal from Hex) — the card's own innate keywords are excluded.</summary>
    public IReadOnlyList<string> AddedKeywords { get; init; } = System.Array.Empty<string>();
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

    /// <summary>Damage per hit for attack intents, after modifiers (enemy Strength/Weak and the
    /// defending player's Vulnerable/etc.); null otherwise.</summary>
    public int? Damage { get; init; }

    /// <summary>The unmodified per-hit damage, so the UI can colour <see cref="Damage"/> when it
    /// differs (more than base = amplified, less = reduced); null otherwise.</summary>
    public int? BaseDamage { get; init; }

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

    /// <summary>The id of the act's boss encounter (names the boss node); null if unknown.</summary>
    public string? BossEncounterId { get; init; }

    /// <summary>The id of the act's second boss encounter in double-boss mode; null otherwise.</summary>
    public string? SecondBossEncounterId { get; init; }
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
