using System;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// End-to-end "play a real game forward" smoke test: from the opening Neow event, a greedy
/// <see cref="AutoPlayer"/> drives the run through the first act's rooms — events, combats (faithful
/// card play with block-then-attack, enemy turns), post-combat rewards, rest sites, treasure, and
/// map navigation — entirely through the public option API. The player is buffed to a huge HP pool
/// so the still-simple combat survives, exercising the room/navigation breadth. The run reaches the
/// act's late section (floor 16), right up to the act-1 boss; entering the boss fight itself is
/// currently blocked by a missing shim type (<c>Godot.GpuParticles2D</c>, used by the
/// CeremonialBeast boss's move) — a shim-growth gap, not a room gap. (The BygoneEffigy elite
/// enemy-turn stall and the AromaOfChaos event-option NRE that used to block this run earlier are
/// now fixed and covered by <see cref="BygoneEffigyTests"/> / <see cref="AromaOfChaosTests"/>.)
/// This guards that a deep multi-floor, multi-room-type run executes through the public API without
/// the harness throwing.
/// </summary>
public sealed class WalkthroughTests
{
    private readonly ITestOutputHelper _out;

    public WalkthroughTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task GreedyRun_NavigatesEventsCombatsRestAndTreasure_DeepIntoTheAct()
    {
        var t = Task.Run(Run);
        await t.WaitAsync(TimeSpan.FromSeconds(180));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Buff the player to a huge HP pool so the (still simple) greedy combat survives the act and
        // we exercise the room/navigation breadth rather than dying to chip damage.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        // Play forward, steering toward the boss, until we are back on the map deep into the act —
        // by which point the run has cleared several combats and passed through rest and treasure
        // rooms, all through the public option API and without the harness throwing.
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Phase == GamePhase.Map && s.Floor >= 16,
            preferMapPointType: MegaCrit.Sts2.Core.Map.MapPointType.Boss,
            log: _out);

        _out.WriteLine($"Run ended: phase={end.Phase} floor={end.Floor} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp} relics=[{string.Join(",", end.Players[0].Relics)}]");

        Assert.Equal(GamePhase.Map, end.Phase);
        Assert.True(end.Floor >= 16, $"expected to reach floor 16+ but stopped on floor {end.Floor}");
        Assert.True(end.Players[0].CurrentHp > 0);
        // Evidence of treasure-room traversal: the starting + Neow relic plus a treasure relic.
        Assert.True(end.Players[0].Relics.Count >= 3,
            $"expected to have picked up a treasure relic but have only {end.Players[0].Relics.Count} relics");
    }
}
