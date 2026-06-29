using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// M7 — determinism. A run is driven entirely from its seed (the master seed derives the named RNG
/// streams), so the same seed plus the same legal inputs must reproduce the same run bit-for-bit.
/// These tests play two independent runs of the same seed forward through the greedy
/// <see cref="AutoPlayer"/> (identical inputs) and assert their observable state matches exactly at a
/// checkpoint — exercising map generation, combat, reward and event RNG along the way — and that a
/// different seed diverges.
/// </summary>
public sealed class DeterminismTests
{
    private readonly ITestOutputHelper _out;

    public DeterminismTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task SameSeed_ReproducesTheSameRun()
    {
        // Run twice (sequentially — one run per process), comparing a full state signature at a
        // checkpoint a few rooms in. Identical seeds + identical greedy play ⇒ identical signature.
        string a = await Task.Run(() => PlayToCheckpoint("DETERMINISM")).WaitAsync(TimeSpan.FromSeconds(120));
        string b = await Task.Run(() => PlayToCheckpoint("DETERMINISM")).WaitAsync(TimeSpan.FromSeconds(120));

        _out.WriteLine("--- run A ---");
        _out.WriteLine(a);
        Assert.Equal(a, b);
    }

    [Fact]
    public async Task DifferentSeeds_Diverge()
    {
        string a = await Task.Run(() => PlayToCheckpoint("SEED_ALPHA")).WaitAsync(TimeSpan.FromSeconds(120));
        string b = await Task.Run(() => PlayToCheckpoint("SEED_BETA")).WaitAsync(TimeSpan.FromSeconds(120));

        Assert.NotEqual(a, b);
    }

    [Fact]
    public async Task Snapshot_Restore_PreservesState_AndIsPlayable()
    {
        await Task.Run(RunSnapshotRestore).WaitAsync(TimeSpan.FromSeconds(120));
    }

    private void RunSnapshotRestore()
    {
        // Play forward to a point on the map after the first combat, snapshot there.
        GameHost host = TestNav.StartOnMap("SNAPSHOT");
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);
        AutoPlayer.Advance(host, stop: s => s.Phase == GamePhase.Reward, maxSteps: 2000);
        // Take the rewards and proceed back to the map (so the snapshot is a clean between-rooms state).
        AutoPlayer.Advance(host, stop: s => s.Phase == GamePhase.Map, maxSteps: 200);

        GameState before = host.GetState();
        string sigBefore = Signature(before);
        _out.WriteLine($"snapshot on map at floor {before.Floor}");
        MegaCrit.Sts2.Core.Saves.SerializableRun save = host.Snapshot();

        // Restore the snapshot into a fresh run; its full observable state — players (HP/gold/deck/
        // relics/potions), act/floor/score and the act's map graph — must match what we snapshotted.
        GameHost restored = GameHost.Restore(save, "SNAPSHOT");
        Assert.Equal(sigBefore, Signature(restored.GetState()));

        // The restored run is live and playable: drive it forward a couple of floors (or to a
        // terminal state) through the public option API without the harness throwing or hanging.
        // (Continued play is not asserted bit-identical to the original: the game deliberately does
        // not persist non-combat RNG across save/load — see MoveToMapCoordAction's own note that it
        // "does not depend on RNGs being deterministic outside of combat" — so the upcoming room-type
        // rolls may legitimately differ. Bit-for-bit reproduction is covered by same-seed replay above.)
        TestNav.SetHp(restored, maxHp: 9999, currentHp: 9999);
        GameState end = AutoPlayer.Advance(
            restored,
            stop: s => s.Floor >= before.Floor + 2 || s.Phase == GamePhase.GameOver,
            maxSteps: 3000);
        Assert.True(end.Floor >= before.Floor + 2 || end.Phase == GamePhase.GameOver,
            $"restored run failed to advance (stopped at phase {end.Phase}, floor {end.Floor})");
    }

    /// <summary>
    /// Play a fresh run of the given seed forward (greedy legal play) until it has climbed a few
    /// floors (or ended), buffing HP so the early combats are survived, then return a deterministic
    /// signature of the resulting state.
    /// </summary>
    private static string PlayToCheckpoint(string seed)
    {
        GameHost host = TestNav.StartOnMap(seed);
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Floor >= 5 || s.Phase == GamePhase.GameOver,
            maxSteps: 4000);
        return Signature(end);
    }

    /// <summary>
    /// A stable, order-independent textual signature of the observable game state: floor/act/score,
    /// each player's HP/gold/sorted deck (with upgrade marks)/relics/potions, the act's map graph,
    /// and any combat roster. Two runs with the same signature reached identical mechanical state.
    /// </summary>
    internal static string Signature(GameState s)
    {
        var sb = new StringBuilder();
        sb.Append($"phase={s.Phase} act={s.ActIndex} floor={s.Floor} score={s.Score} victory={s.IsVictory}\n");

        foreach (PlayerState p in s.Players)
        {
            sb.Append($"player {p.NetId} {p.Character} hp={p.CurrentHp}/{p.MaxHp} gold={p.Gold} energy={p.MaxEnergy}\n");
            sb.Append("  deck=[")
              .Append(string.Join(",", p.Deck.Select(CardSig).OrderBy(x => x, StringComparer.Ordinal)))
              .Append("]\n");
            sb.Append("  relics=[")
              .Append(string.Join(",", p.Relics.OrderBy(x => x, StringComparer.Ordinal)))
              .Append("]\n");
            sb.Append("  potions=[")
              .Append(string.Join(",", p.Potions.Select(x => x ?? "_")))
              .Append("]\n");
        }

        if (s.Map is { } map)
        {
            sb.Append($"map act={map.ActIndex} at={map.CurrentCoord}\n");
            foreach (MapPointView pt in map.Points.OrderBy(p => (p.Coord.Col, p.Coord.Row)))
            {
                string children = string.Join("|", pt.Children
                    .OrderBy(c => (c.Col, c.Row)).Select(c => $"{c.Col},{c.Row}"));
                sb.Append($"  ({pt.Coord.Col},{pt.Coord.Row})={pt.PointType}->{children}\n");
            }
            sb.Append("  reachable=[")
              .Append(string.Join(",", map.Reachable.OrderBy(c => (c.Col, c.Row)).Select(c => $"{c.Col},{c.Row}")))
              .Append("]\n");
        }

        // Only signature the combat roster when actually in combat — out of combat the projection can
        // still echo the just-ended fight's stale state (CombatManager.DebugOnlyGetState), which is
        // not part of the run's persistent state and would spuriously differ across a restore.
        if (s.Phase == GamePhase.Combat && s.Combat is { } combat)
        {
            sb.Append($"combat round={combat.RoundNumber} side={combat.CurrentSide}\n");
            foreach (EnemyView e in combat.Enemies.OrderBy(e => e.CombatId))
            {
                sb.Append($"  enemy {e.MonsterId} hp={e.CurrentHp}/{e.MaxHp} block={e.Block}\n");
            }
        }

        return sb.ToString();
    }

    private static string CardSig(CardView c) => c.CardId + (c.Upgraded ? "+" : string.Empty);
}
