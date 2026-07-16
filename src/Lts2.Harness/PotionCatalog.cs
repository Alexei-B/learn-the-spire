using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Factories;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Potions;

namespace Lts2.Harness;

/// <summary>
/// Single source of truth for which potions a training scenario may grant, and — critically — which it
/// must never grant because using them restores or grants the player HP. Combat rewards are HP-based
/// (see <see cref="CombatScenario"/>'s <c>hpLost</c>), so an HP-granting potion in the belt would let the
/// policy recover reward "for free" and skew training; the realistic deck spec therefore excludes them.
/// </summary>
public static class PotionCatalog
{
    /// <summary>
    /// The potion model types excluded from a realistic grant because their effect <b>restores or grants
    /// player HP</b> (or revives). Derived by <em>mechanically inspecting every potion's <c>OnUse</c>
    /// effect</em> in the decompile (<c>refsrc/…/Models.Potions</c>) — the only potions whose effect
    /// reaches <c>CreatureCmd.Heal</c> / <c>CreatureCmd.GainMaxHp</c> or a healing power:
    /// <list type="bullet">
    /// <item><description><see cref="BloodPotion"/> — <c>CreatureCmd.Heal</c>, heals 20% of max HP.</description></item>
    /// <item><description><see cref="FruitJuice"/> — <c>CreatureCmd.GainMaxHp</c>, permanently +5 max HP.</description></item>
    /// <item><description><see cref="RegenPotion"/> — applies <c>RegenPower</c>, which heals at each of the
    ///   owner's turn ends.</description></item>
    /// <item><description><see cref="FairyInABottle"/> — <c>CreatureCmd.Heal</c> and revives the owner on a
    ///   lethal hit (<c>AfterPreventingDeath</c>).</description></item>
    /// </list>
    /// The game's own <see cref="PotionModel.CanBeGeneratedInCombat"/><c> == false</c> flag (documented as
    /// "can heal the player … or do other restricted actions") covers three of these — FruitJuice, RegenPotion,
    /// FairyInABottle — but <b>not</b> BloodPotion, so it is insufficient on its own. We therefore union the
    /// curated heal-effect set with that flag: the curated set is authoritative, and the flag is a
    /// forward-compatible safety net for any future potion the game itself marks unfit for free generation.
    /// (Today the union equals exactly the four types above.)
    /// </summary>
    private static readonly HashSet<Type> HpTypes = new()
    {
        typeof(BloodPotion),
        typeof(FruitJuice),
        typeof(RegenPotion),
        typeof(FairyInABottle),
    };

    /// <summary>The excluded potion model types (for tests / diagnostics).</summary>
    public static IReadOnlyCollection<Type> HpExcludedTypes => HpTypes;

    /// <summary>True if <paramref name="potion"/> restores or grants player HP (or revives) and so must
    /// never be granted to a realistic scenario. Mechanical: the curated heal-effect type set unioned with
    /// the game's own <see cref="PotionModel.CanBeGeneratedInCombat"/> flag.</summary>
    public static bool RestoresOrGrantsHp(PotionModel potion)
    {
        if (potion is null)
        {
            throw new ArgumentNullException(nameof(potion));
        }
        return HpTypes.Contains(potion.GetType()) || !potion.CanBeGeneratedInCombat;
    }

    /// <summary>
    /// The potions a realistic scenario may grant to <paramref name="player"/>: the player's own character
    /// pool plus the shared pool (the faithful reward universe from <see cref="PotionFactory.GetPotionOptions"/>),
    /// minus every HP-restoring/granting potion. Canonical models, distinct, in a deterministic order.
    /// </summary>
    public static IReadOnlyList<PotionModel> GrantablePool(Player player)
    {
        if (player is null)
        {
            throw new ArgumentNullException(nameof(player));
        }
        return PotionFactory.GetPotionOptions(player, Array.Empty<PotionModel>())
            .Where(p => !RestoresOrGrantsHp(p))
            .Distinct()
            .ToList();
    }

    /// <summary>The ids of the HP-restoring/granting potions excluded from a realistic grant, computed over
    /// every registered potion — for the deck-spec docs, diagnostics, and the "the filter is non-empty"
    /// invariant. Sorted for stable output.</summary>
    public static IReadOnlyList<string> HpExcludedPotionIds() =>
        ModelDb.AllPotions
            .Where(RestoresOrGrantsHp)
            .Select(p => p.Id.Entry)
            .Distinct()
            .OrderBy(id => id, StringComparer.Ordinal)
            .ToList();
}
