using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Map;
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
        throw new InvalidOperationException($"AutoPlayer exceeded {maxSteps} steps without stopping.");
    }

    private static void StepCombat(GameHost host)
    {
        var options = host.ListOptions();
        var plays = options.Where(o => o.Kind == OptionKind.PlayCard).ToList();
        if (plays.Count == 0)
        {
            host.Apply(options.First(o => o.Kind == OptionKind.EndTurn));
            return;
        }

        // Focus-fire: among the playable cards, prefer an attack on the lowest-HP enemy so
        // enemies die sooner and incoming damage drops. Non-targeted cards (block/buff) sort
        // last but are still all played over successive steps before the turn ends.
        var enemyHp = host.Combat?.Enemies.ToDictionary(e => e.CombatId ?? 0u, e => e.CurrentHp)
            ?? new Dictionary<uint, int>();
        GameOption best = plays
            .OrderBy(o => o.TargetCombatId is { } id && enemyHp.TryGetValue(id, out int hp) ? hp : int.MaxValue)
            .First();
        host.Apply(best);
    }

    private static void StepReward(GameHost host)
    {
        var options = host.ListOptions();
        GameOption? take = options.FirstOrDefault(o => o.Kind == OptionKind.TakeReward);
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
