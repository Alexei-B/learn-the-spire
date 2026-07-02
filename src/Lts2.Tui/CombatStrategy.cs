using System.Collections.Generic;
using System.Linq;
using Lts2.Harness;
using MegaCrit.Sts2.Core.Entities.Cards;

namespace Lts2.Tui;

/// <summary>
/// The TUI's default combat policy — the move the Tab auto-play shortcut takes. Pure over the harness
/// read-model (<see cref="GameState"/>) and the listed <see cref="GameOption"/>s, so it holds no live
/// game references and is easy to reason about. The policy, in priority order:
///
/// <list type="number">
///   <item>If you would take damage this turn (projected incoming &gt; current block): consider your
///   block cards from the most block-per-energy down, and play the first whose block would be at least
///   80% used (not mostly wasted on overblock). If none qualify, don't block.</item>
///   <item>Otherwise, with attack cards in hand: from the cheapest up, if any attack would kill a
///   target, play it against that target; otherwise play the highest damage-per-energy attack against
///   the lowest-health enemy.</item>
///   <item>Otherwise play a power card.</item>
///   <item>Otherwise play a skill card.</item>
///   <item>Otherwise, if you are out of energy with no zero-cost cards to play, end the turn.</item>
/// </list>
///
/// Returns null when there is no move to make (not in combat, or you still have plays available that
/// the policy doesn't act on — the caller decides what to do then).
/// </summary>
internal static class CombatStrategy
{
    /// <summary>
    /// Choose the default combat move for <paramref name="state"/> from <paramref name="options"/>
    /// (as returned by <see cref="GameHost.ListOptions(ulong)"/>), or null if the policy has no move.
    /// </summary>
    public static GameOption? ChooseDefaultMove(GameState state, IReadOnlyList<GameOption> options)
    {
        if (state.Phase != GamePhase.Combat || state.Combat is not { } combat || state.Players.Count == 0)
        {
            return null;
        }
        PlayerState me = state.Players[0];
        int energy = me.CombatState?.Energy ?? 0;

        List<GameOption> playable = options
            .Where(o => o.Kind == OptionKind.PlayCard && o.Card is not null)
            .ToList();

        // 1. Defensive: if we'd take damage this turn, play the best-value defensive card (if efficient).
        // "Defensive value" is printed block plus character block-substitutes (the Necrobinder's Osty
        // summon soaks hits like block — see EffectiveBlock). Rank by value-per-energy; on a tie, favour
        // the summon (Osty persists across turns, so it's the better spend for equal value).
        // A live Osty already soaks incoming attacks: an enemy's (powered) attack that gets past our block
        // is redirected onto Osty and only its overkill — damage beyond Osty's current HP — spills onto us
        // (Osty's own block is bypassed on the redirect). So Osty's HP is a buffer on top of our block.
        int ostyBuffer = me.CombatState?.Osty is { IsAlive: true } osty ? osty.CurrentHp : 0;
        int unblocked = IncomingDamage(combat) - me.Block - ostyBuffer;
        if (unblocked > 0)
        {
            IEnumerable<GameOption> blockCards = playable
                .Where(o => EffectiveBlock(o.Card!) > 0)
                .OrderByDescending(o => PerEnergy(EffectiveBlock(o.Card!), EffectiveCost(o.Card!, energy)))
                .ThenByDescending(o => o.Card!.Summon ?? 0);
            foreach (GameOption o in blockCards)
            {
                int block = EffectiveBlock(o.Card!);
                // At least 80% of the block absorbs real damage: min(block, unblocked) >= 0.8 * block,
                // i.e. unblocked >= 0.8 * block. Integer form avoids float rounding.
                if (unblocked * 5 >= block * 4)
                {
                    return o;
                }
            }
            // No defensive card is worth playing — fall through and attack instead.
        }

        // 2. Offensive: attack cards.
        List<GameOption> attacks = playable.Where(o => o.Card!.Type == CardType.Attack).ToList();
        if (attacks.Count > 0)
        {
            Dictionary<uint, EnemyView> enemies = combat.Enemies
                .Where(e => e.IsHittable && e.CurrentHp > 0)
                .ToDictionary(e => e.CombatId);

            // 2a. From the cheapest attack up, take a lethal hit if one is available.
            GameOption? lethal = attacks
                .Where(o => o.TargetCombatId is { } id
                            && enemies.TryGetValue(id, out EnemyView? e)
                            && (o.Card!.Damage ?? 0) >= e!.CurrentHp + e.Block)
                .OrderBy(o => EffectiveCost(o.Card!, energy))
                .ThenBy(o => enemies[o.TargetCombatId!.Value].CurrentHp)
                .FirstOrDefault();
            if (lethal is not null)
            {
                return lethal;
            }

            // 2b. Otherwise the highest damage-per-energy attack, aimed at the lowest-health enemy.
            EnemyView? weakest = enemies.Values.OrderBy(e => e.CurrentHp).FirstOrDefault();
            GameOption? best = attacks
                .Where(o => o.TargetCombatId is null
                            || (weakest is not null && o.TargetCombatId == weakest.CombatId))
                .OrderByDescending(o => PerEnergy(o.Card!.Damage ?? 0, EffectiveCost(o.Card!, energy)))
                .FirstOrDefault();
            // If the weakest enemy can't be targeted by any attack, fall back to any attack.
            return best ?? attacks
                .OrderByDescending(o => PerEnergy(o.Card!.Damage ?? 0, EffectiveCost(o.Card!, energy)))
                .First();
        }

        // 3. A power card.
        GameOption? power = playable.FirstOrDefault(o => o.Card!.Type == CardType.Power);
        if (power is not null)
        {
            return power;
        }

        // 4. A skill card.
        GameOption? skill = playable.FirstOrDefault(o => o.Card!.Type == CardType.Skill);
        if (skill is not null)
        {
            return skill;
        }

        // 5. Nothing worth playing — end the turn. Reached when there's no attack/power/skill and no
        // efficient block to play: either you're out of energy, or the only cards left are ones the
        // policy doesn't act on (clogging statuses/curses) even though you still have energy to spend.
        return options.FirstOrDefault(o => o.Kind == OptionKind.EndTurn);
    }

    /// <summary>
    /// A card's block-equivalent defensive value: printed block plus character block-substitutes the
    /// policy weighs like block — currently the Necrobinder's Osty summon (a wall that soaks hits).
    /// (Defect orb block and Silent sly-on-discard are not modelled here yet.)
    /// </summary>
    private static int EffectiveBlock(CardView card) => (card.Block ?? 0) + (card.Summon ?? 0);

    /// <summary>The card's energy cost for ranking: an X-cost card spends all current energy.</summary>
    private static int EffectiveCost(CardView card, int energy) => card.CostsX ? energy : card.EnergyCost;

    /// <summary>A value-per-energy ratio for ranking; a free (0-cost) card ranks as infinitely efficient
    /// when it does anything, and as zero when it doesn't.</summary>
    private static double PerEnergy(int value, int cost) =>
        cost <= 0 ? (value > 0 ? double.PositiveInfinity : 0.0) : (double)value / cost;

    /// <summary>Total projected damage the enemies' telegraphed attack intents would deal this turn,
    /// before the player's block (each intent's per-hit damage is already after all modifiers).</summary>
    private static int IncomingDamage(CombatView combat)
    {
        int total = 0;
        foreach (EnemyView e in combat.Enemies)
        {
            foreach (IntentView i in e.Intents)
            {
                if (i.Damage is { } d)
                {
                    total += d * (i.Hits ?? 1);
                }
            }
        }
        return total;
    }
}
