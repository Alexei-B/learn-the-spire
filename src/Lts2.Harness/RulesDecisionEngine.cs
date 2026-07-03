using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;

namespace Lts2.Harness;

/// <summary>
/// A simple hand-written combat policy expressed as an <see cref="IDecisionEngine"/> — the default the
/// TUI's "auto-play" shortcut uses. It scores the legal combat options in priority bands so that the
/// top-scored option is the move the policy would take; the full scored list is a coarse ranking useful
/// as a training/eval baseline. The policy, in descending priority:
///
/// <list type="number">
///   <item><b>Efficient block</b> — if you would take damage this turn (projected incoming &gt; your
///   block plus any Osty buffer), the block cards, best block-per-energy first, scored above everything
///   else, but <em>only</em> when at least 80% of the block would absorb real damage (not overblock).</item>
///   <item><b>Lethal attack</b> — the cheapest attack that would kill its target (ties: the
///   lowest-health target).</item>
///   <item><b>Best attack</b> — the highest damage-per-energy attack aimed at the lowest-health enemy
///   (or an untargeted hit); other targets rank below.</item>
///   <item><b>Power</b>, then <b>Skill</b> cards.</item>
///   <item><b>End turn</b> — scored just above cards the policy won't play (statuses/curses), so the turn
///   ends only when nothing worth playing remains.</item>
/// </list>
///
/// Only combat is scored; off the battlefield <see cref="Evaluate"/> returns an empty list (the engine
/// declines). This is a faithful port of the policy that previously lived in the TUI: the top-scored
/// option matches the old <c>ChooseDefaultMove</c> pick.
/// </summary>
public sealed class RulesDecisionEngine : IDecisionEngine
{
    // Priority bands. Each band strictly dominates the ones below it; within-band deltas (per-energy
    // value, cost, target health, summon tie-break) stay well under the band gap so they never cross
    // bands, which keeps the top-scored option equal to the old priority-ordered policy.
    private const double BlockBand = 500.0;   // an efficient defensive play
    private const double LethalBand = 400.0;  // an attack that secures a kill
    private const double AttackBand = 300.0;  // damage-per-energy at the weakest enemy
    private const double PowerBand = 200.0;
    private const double SkillBand = 100.0;
    private const double EndTurnBand = 0.0;   // baseline — beaten by any card worth playing
    private const double JunkBand = -1.0;     // a playable-but-worthless card (status/curse): below End Turn
    private const int CostCap = 30;           // clamps a cost term so it can't reach the next band

    public string Name => "Rules";

    public IReadOnlyList<ScoredOption> Evaluate(GameState state, IReadOnlyList<GameOption> options)
    {
        var scored = new List<ScoredOption>();
        if (state.Phase != GamePhase.Combat || state.Combat is not { } combat || state.Players.Count == 0)
        {
            return scored; // combat-only policy: no opinion elsewhere.
        }

        PlayerState me = state.Players[0];
        int energy = me.CombatState?.Energy ?? 0;

        // A live Osty soaks incoming (powered) attacks: only damage beyond its current HP spills onto the
        // player, so its HP is a buffer on top of our block (see the defensive branch in the design notes).
        int ostyBuffer = me.CombatState?.Osty is { IsAlive: true } osty ? osty.CurrentHp : 0;
        int unblocked = IncomingDamage(combat) - me.Block - ostyBuffer;

        Dictionary<uint, EnemyView> enemies = combat.Enemies
            .Where(e => e.IsHittable && e.CurrentHp > 0)
            .ToDictionary(e => e.CombatId);
        EnemyView? weakest = enemies.Values.OrderBy(e => e.CurrentHp).FirstOrDefault();

        foreach (GameOption o in options)
        {
            if (o.Kind == OptionKind.EndTurn)
            {
                scored.Add(new ScoredOption(o, EndTurnBand, "End the turn."));
            }
            else if (o.Kind == OptionKind.PlayCard && o.Card is { } card)
            {
                (double score, string why) = ScoreCard(o, card, energy, unblocked, enemies, weakest);
                scored.Add(new ScoredOption(o, score, why));
            }
            // Other option kinds (e.g. potion use) are outside this policy — left unscored.
        }
        return scored;
    }

    /// <summary>Score one card play as the best of the roles it can fill (block / lethal / attack / power / skill).</summary>
    private static (double score, string why) ScoreCard(
        GameOption o, CardView card, int energy, int unblocked,
        Dictionary<uint, EnemyView> enemies, EnemyView? weakest)
    {
        double best = JunkBand;
        string why = "No useful effect this turn.";
        int cost = EffectiveCost(card, energy);

        // 1. Defensive: a block card is worth playing only when we'd take damage and at least 80% of the
        //    block absorbs it (unblocked >= 0.8 * block, in integer form). Best block-per-energy first,
        //    with the Osty summon (which persists) favoured on a tie.
        int effBlock = EffectiveBlock(card);
        if (unblocked > 0 && effBlock > 0 && unblocked * 5 >= effBlock * 4)
        {
            double s = BlockBand + Squash(PerEnergy(effBlock, cost)) + System.Math.Min(card.Summon ?? 0, 50) * 1e-4;
            (best, why) = Better(best, why, s, $"Efficient block ({effBlock}).");
        }

        if (card.Type == CardType.Attack)
        {
            int dmg = card.Damage ?? 0;

            // 2. Lethal: this option targets an enemy it would kill (through its block). Cheapest first,
            //    then the lowest-health target.
            if (o.TargetCombatId is { } id && enemies.TryGetValue(id, out EnemyView? e)
                && dmg >= e!.CurrentHp + e.Block)
            {
                double s = LethalBand + (CostCap - System.Math.Min(cost, CostCap)) + (1.0 - Squash(e.CurrentHp)) * 0.01;
                (best, why) = Better(best, why, s, $"Lethal on {e.MonsterId}.");
            }

            // 3. Best damage-per-energy. An attack aimed at the lowest-health enemy (or untargeted) ranks
            //    above one aimed elsewhere, so the fallback "any attack" only wins when nothing hits the
            //    weakest enemy.
            bool atWeakest = o.TargetCombatId is null || (weakest is not null && o.TargetCombatId == weakest.CombatId);
            double dpe = Squash(PerEnergy(dmg, cost));
            double atkScore = AttackBand + (atWeakest ? 0.5 : 0.0) + dpe * 0.4;
            (best, why) = Better(best, why, atkScore,
                atWeakest ? "Best damage on the weakest enemy." : "Attack (no shot at the weakest enemy).");
        }
        else if (card.Type == CardType.Power)
        {
            (best, why) = Better(best, why, PowerBand, "Play a power.");
        }
        else if (card.Type == CardType.Skill)
        {
            (best, why) = Better(best, why, SkillBand, "Play a skill.");
        }

        return (best, why);
    }

    private static (double, string) Better(double best, string why, double candidate, string candidateWhy) =>
        candidate > best ? (candidate, candidateWhy) : (best, why);

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

    /// <summary>Map a (possibly infinite) per-energy ratio to a strictly-increasing value in [0, 1] — so a
    /// higher ratio always scores higher within its band, and a free effective play (+∞) tops the band.</summary>
    private static double Squash(double ratio) =>
        double.IsPositiveInfinity(ratio) ? 1.0 : ratio / (ratio + 1.0);

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
