using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using Sts2.Harness;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// A greedy, fully-legal driver that advances a run through the public option API. It makes a
/// single legal choice per step (play a card, end turn, take a reward, pick an event/treasure
/// option, move on the map) so callers can run a real game forward until some condition holds.
/// Used to reach later rooms (combats, events, treasure) the way a normal playthrough would.
/// </summary>
internal static class AutoPlayer
{
    /// <summary>
    /// Advance the run until <paramref name="stop"/> returns true, the run ends, or the driver
    /// gets stuck on a room it cannot model. Returns the state it stopped on.
    /// </summary>
    /// <param name="preferMapPointType">
    /// When set, map moves steer toward the nearest reachable point of this type (e.g.
    /// <see cref="MapPointType.Treasure"/>); otherwise the leftmost reachable point is taken.
    /// </param>
    public static GameState Advance(
        GameHost host,
        Func<GameState, bool> stop,
        MapPointType? preferMapPointType = null,
        int maxSteps = 5000,
        ITestOutputHelper? log = null)
    {
        GamePhase lastPhase = (GamePhase)(-1);
        for (int step = 0; step < maxSteps; step++)
        {
            GameState s = host.GetState();
            if (stop(s))
            {
                return s;
            }
            if (log is not null && s.Phase != lastPhase)
            {
                lastPhase = s.Phase;
                log.WriteLine($"  [{s.Phase}] floor={s.Floor} room={host.Run.CurrentRoom?.GetType().Name} hp={s.Players[0].CurrentHp}/{s.Players[0].MaxHp}");
            }

            switch (s.Phase)
            {
                case GamePhase.Combat:
                    StepCombat(host);
                    break;
                case GamePhase.Choice:
                    // Resolve a mid-effect card choice with the first offered selection.
                    host.Apply(host.ListOptions().First());
                    break;
                case GamePhase.Reward:
                    StepReward(host);
                    break;
                case GamePhase.Event:
                    host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.ChooseEventOption));
                    break;
                case GamePhase.Treasure:
                    // Default: take the first relic on offer.
                    host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.TakeTreasureRelic));
                    break;
                case GamePhase.RestSite:
                    StepRest(host);
                    break;
                case GamePhase.Shop:
                    StepShop(host);
                    break;
                case GamePhase.CrystalSphere:
                    StepCrystalSphere(host, s);
                    break;
                case GamePhase.Map:
                    StepMap(host, s, preferMapPointType, log);
                    break;
                case GamePhase.GameOver:
                    return s;
                default:
                    // An unmodelled room/screen: nothing legal to apply.
                    log?.WriteLine($"AutoPlayer stuck at phase {s.Phase} room={host.Run.CurrentRoom?.GetType().Name}");
                    return s;
            }
        }
        if (log is not null)
        {
            GameState s = host.GetState();
            log.WriteLine($"BUDGET EXCEEDED at phase={s.Phase} floor={s.Floor} room={host.Run.CurrentRoom?.GetType().Name}");
            if (s.Rewards is { } rv)
            {
                foreach (RewardView r in rv.Rewards)
                {
                    log.WriteLine($"  reward {r.Type} taken={r.Taken} gold={r.Gold} potion={r.PotionId} relic={r.RelicId} cards={r.Cards?.Count}");
                }
                log.WriteLine($"  potions=[{string.Join(",", s.Players[0].Potions)}]");
            }
            if (s.Combat is { } c)
            {
                foreach (EnemyView e in c.Enemies)
                {
                    string intents = string.Join(",", e.Intents.Select(i => $"{i.Type}({i.Damage}x{i.Hits})"));
                    log.WriteLine($"  enemy {e.MonsterId} hp={e.CurrentHp}/{e.MaxHp} block={e.Block} hittable={e.IsHittable} intents=[{intents}]");
                }
                var pcs = s.Players[0].CombatState;
                if (pcs is not null)
                {
                    log.WriteLine($"  energy={pcs.Energy}/{pcs.MaxEnergy} hand=[{string.Join(",", pcs.Hand.Select(h => $"{h.CardId}{(h.CanPlay ? "" : "*")}"))}]");
                }
            }
        }
        throw new InvalidOperationException($"AutoPlayer exceeded {maxSteps} steps without stopping.");
    }

    /// <summary>
    /// Play one combat action. Strategy: block while a block card still has full utility (incoming
    /// damage this turn exceeds the block we already have), then focus-fire attacks on the
    /// lowest-HP enemy, then play any remaining utility, then end the turn. Works directly off the
    /// live combat state (and faithful play via <see cref="GameHost.PlayCard"/>); mid-combat card
    /// choices still surface to the outer loop as <see cref="GamePhase.Choice"/>.
    /// </summary>
    private static void StepCombat(GameHost host)
    {
        var combat = host.Combat!;
        Player player = combat.Players.Single();
        var pcs = player.PlayerCombatState!;
        var playable = pcs.Hand.Cards.Where(c => c.CanPlay(out _, out _)).ToList();
        if (playable.Count == 0)
        {
            host.EndTurn(player);
            return;
        }

        int incoming = IncomingDamage(host);

        // 1) Block while it pays off in full (we are still taking more than we can block).
        if (player.Creature.Block < incoming)
        {
            CardModel? blockCard = playable.FirstOrDefault(c => c.GainsBlock);
            if (blockCard is not null && host.PlayCard(blockCard, null))
            {
                return;
            }
        }

        // 2) Attack the lowest-HP hittable enemy (kill things fast to cut incoming damage).
        CardModel? attack = playable.FirstOrDefault(c => c.TargetType == TargetType.AnyEnemy);
        if (attack is not null)
        {
            Creature? target = combat.HittableEnemies.OrderBy(e => e.CurrentHp).FirstOrDefault();
            if (target is not null && attack.IsValidTarget(target) && host.PlayCard(attack, target))
            {
                return;
            }
        }

        // 3) Any other playable card (untargeted buffs/powers, or surplus block once safe).
        CardModel? other = playable.FirstOrDefault(c => c.TargetType != TargetType.AnyEnemy);
        if (other is not null && host.PlayCard(other, null))
        {
            return;
        }

        host.EndTurn(player);
    }

    /// <summary>Total damage the enemies' telegraphed attack intents would deal this turn.</summary>
    private static int IncomingDamage(GameHost host)
    {
        CombatView? combat = host.GetState().Combat;
        if (combat is null)
        {
            return 0;
        }
        int total = 0;
        foreach (EnemyView enemy in combat.Enemies)
        {
            foreach (IntentView intent in enemy.Intents)
            {
                if (intent.Damage is int dmg)
                {
                    total += dmg * (intent.Hits ?? 1);
                }
            }
        }
        return total;
    }

    /// <summary>
    /// Spend a Crystal Sphere divination. Strategy: click the hidden cell whose surrounding 3×3
    /// (the Big tool's footprint) covers the most still-hidden cells belonging to a not-yet-revealed
    /// item, so divinations make progress toward fully uncovering items (which is what pays out).
    /// Falls back to any hidden cell.
    /// </summary>
    private static void StepCrystalSphere(GameHost host, GameState s)
    {
        CrystalSphereView view = s.CrystalSphere!;
        var hidden = new HashSet<Coord>(view.HiddenCells);

        // Cells that still hide a not-yet-revealed item — uncovering these is what earns rewards.
        var itemCells = new HashSet<Coord>();
        foreach (CrystalSphereItemView item in view.Items)
        {
            if (item.Revealed)
            {
                continue;
            }
            for (int i = 0; i < item.Size.Col; i++)
            {
                for (int j = 0; j < item.Size.Row; j++)
                {
                    var c = new Coord(item.Position.Col + i, item.Position.Row + j);
                    if (hidden.Contains(c))
                    {
                        itemCells.Add(c);
                    }
                }
            }
        }

        var options = host.ListOptions().Where(o => o.Kind == OptionKind.ClickCrystalSphereCell).ToList();
        GameOption best = options[0];
        int bestCovered = -1;
        foreach (GameOption opt in options)
        {
            Coord cell = opt.CrystalSphereCell!.Value;
            int covered = 0;
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    if (itemCells.Contains(new Coord(cell.Col + dx, cell.Row + dy)))
                    {
                        covered++;
                    }
                }
            }
            if (covered > bestCovered)
            {
                bestCovered = covered;
                best = opt;
            }
        }
        host.Apply(best);
    }

    private static void StepRest(GameHost host)
    {
        var options = host.ListOptions();
        PlayerState player = host.GetState().Players[0];
        // Rest to heal when hurt; otherwise smith (upgrade a card); otherwise whatever is offered.
        GameOption? heal = options.FirstOrDefault(o => o.RestOptionId == "HEAL");
        if (heal is not null && player.CurrentHp < player.MaxHp)
        {
            host.Apply(heal);
            return;
        }
        GameOption? smith = options.FirstOrDefault(o => o.RestOptionId == "SMITH");
        host.Apply(smith ?? options.First());
    }

    /// <summary>
    /// Shop strategy: greedily buy the first affordable item (each purchase strictly spends gold, so
    /// this terminates — affordable items run out), then leave by moving on. Card removal raises a
    /// card choice that the outer loop resolves as <see cref="GamePhase.Choice"/>.
    /// </summary>
    private static void StepShop(GameHost host)
    {
        var options = host.ListOptions();
        GameOption? buy = options.FirstOrDefault(o => o.Kind == OptionKind.BuyShopItem);
        host.Apply(buy ?? options.First(o => o.Kind == OptionKind.MoveTo));
    }

    private static void StepReward(GameHost host)
    {
        var options = host.ListOptions();
        bool potionSlotFree = host.GetState().Players[0].Potions.Any(p => p is null);

        // Take card/relic/gold rewards; only take a potion when there's a free slot (taking one
        // with full slots is a no-op and would otherwise loop). When nothing is takeable, proceed.
        GameOption? take = options.FirstOrDefault(o =>
            o.Kind == OptionKind.TakeReward
            && (potionSlotFree || !o.Description.StartsWith("Take potion", StringComparison.Ordinal)));
        host.Apply(take ?? options.First(o => o.Kind == OptionKind.ProceedFromRewards));
    }

    private static void StepMap(GameHost host, GameState s, MapPointType? prefer, ITestOutputHelper? log)
    {
        var moves = host.ListOptions().Where(o => o.Kind == OptionKind.MoveTo).ToList();
        if (moves.Count == 0)
        {
            throw new InvalidOperationException("On the map with no moves available.");
        }

        GameOption chosen = moves[0];
        if (prefer is { } want && s.Map is { } map)
        {
            GameOption? steered = SteerToward(map, moves, want);
            if (steered is not null)
            {
                chosen = steered;
            }
        }
        log?.WriteLine($"Move to {chosen.Coord} (floor {s.Floor})");
        host.Apply(chosen);
    }

    /// <summary>
    /// Among the reachable moves, pick the one whose map subtree reaches a point of
    /// <paramref name="want"/> in the fewest steps. Returns null if none can.
    /// </summary>
    private static GameOption? SteerToward(MapView map, List<GameOption> moves, MapPointType want)
    {
        var byCoord = map.Points.ToDictionary(p => p.Coord);
        GameOption? best = null;
        int bestDist = int.MaxValue;
        foreach (GameOption move in moves)
        {
            if (move.Coord is not { } start || !byCoord.ContainsKey(start))
            {
                continue;
            }
            int dist = DistanceToType(byCoord, start, want);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = move;
            }
        }
        return best;
    }

    private static int DistanceToType(
        IReadOnlyDictionary<Coord, MapPointView> byCoord, Coord start, MapPointType want)
    {
        var seen = new HashSet<Coord>();
        var queue = new Queue<(Coord coord, int dist)>();
        queue.Enqueue((start, 0));
        seen.Add(start);
        while (queue.Count > 0)
        {
            (Coord coord, int dist) = queue.Dequeue();
            if (!byCoord.TryGetValue(coord, out MapPointView? point))
            {
                continue;
            }
            if (point.PointType == want)
            {
                return dist;
            }
            foreach (Coord child in point.Children)
            {
                if (seen.Add(child))
                {
                    queue.Enqueue((child, dist + 1));
                }
            }
        }
        return int.MaxValue;
    }
}
