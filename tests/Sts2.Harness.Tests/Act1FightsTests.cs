using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Enumerates every act-1 fight (the discrete encounter list of each index-0 act) and drives it to
/// resolution through the public option API, to catch fights that NRE or lock up. Each case starts
/// a fresh run, resolves the opening Neow event, buffs the player to a huge HP pool (so the simple
/// greedy combat survives and we exercise the encounter's mechanics rather than dying to chip
/// damage), then enters the encounter directly (bypassing map navigation) and plays it out — block,
/// focus-fire, end turn — until combat ends and the player is back on the map. A fight "works and
/// resolves" if the harness drives it from start to post-combat rewards without throwing.
///
/// Act 1 today means the two index-0 acts: <c>Overgrowth</c> (the default) and <c>Underdocks</c>.
/// Bosses are included; some still surface missing shim types (see the roadmap M4 notes).
/// </summary>
public sealed class Act1FightsTests
{
    private readonly ITestOutputHelper _out;

    public Act1FightsTests(ITestOutputHelper output) => _out = output;

    // The discrete list of Overgrowth (default act 1) encounters: normals/weaks, elites, bosses.
    public static IEnumerable<object[]> OvergrowthFights => Cases(
        // Normal + weak monster encounters
        "CubexConstructNormal", "FlyconidNormal", "FogmogNormal", "FuzzyWurmCrawlerWeak",
        "InkletsNormal", "MawlerNormal", "NibbitsNormal", "NibbitsWeak", "OvergrowthCrawlers",
        "RubyRaidersNormal", "ShrinkerBeetleWeak", "SlimesNormal", "SlimesWeak",
        "SlitheringStranglerNormal", "SnappingJaxfruitNormal", "VineShamblerNormal",
        // Elites
        "BygoneEffigyElite", "ByrdonisElite", "PhrogParasiteElite",
        // Bosses
        "VantomBoss", "CeremonialBeastBoss", "TheKinBoss");

    // The discrete list of Underdocks (the alternate index-0 act) encounters.
    public static IEnumerable<object[]> UnderdocksFights => Cases(
        // Normal + weak monster encounters
        "CorpseSlugsNormal", "CorpseSlugsWeak", "CultistsNormal", "FossilStalkerNormal",
        "GremlinMercNormal", "HauntedShipNormal", "LivingFogNormal", "PunchConstructNormal",
        "SeapunkNormal", "SeapunkWeak", "SewerClamNormal", "SludgeSpinnerWeak", "ToadpolesWeak",
        "TwoTailedRatsNormal",
        // Elites
        "PhantasmalGardenersElite", "SkulkingColonyElite", "TerrorEelElite",
        // Bosses
        "LagavulinMatriarchBoss", "SoulFyshBoss", "WaterfallGiantBoss");

    [Theory]
    [MemberData(nameof(OvergrowthFights))]
    public async Task OvergrowthFight_Resolves(string encounterName)
    {
        var t = Task.Run(() => RunFight(encounterName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    [Theory]
    [MemberData(nameof(UnderdocksFights))]
    public async Task UnderdocksFight_Resolves(string encounterName)
    {
        var t = Task.Run(() => RunFight(encounterName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunFight(string encounterName)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Buff to a huge HP pool so the greedy combat heuristic survives the act-1 fight and we
        // exercise the encounter's mechanics rather than dying to chip damage.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        EncounterModel encounter = ResolveEncounter(encounterName);
        _out.WriteLine($"Entering {encounterName} ({encounter.RoomType})");
        host.EnterEncounterDebug(encounter);

        Assert.True(host.InCombat, $"expected to be in combat after entering {encounterName}");

        // Play the fight (and its post-combat rewards) out through the public option API until it
        // reaches a terminal state — back on the map (won) or game over (died). A fight "resolves"
        // when the harness drives it to that terminal state without throwing or hanging; reaching it
        // is what this test guards. (The player is buffed to 9999 HP so most fights are won, but a
        // few bosses — e.g. the Lagavulin Matriarch, which drains Strength/Dexterity every cycle —
        // legitimately out-scale the simple greedy AutoPlayer and end in a survivable game over.)
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
            maxSteps: 3000,
            log: _out);

        _out.WriteLine($"{encounterName} ended: phase={end.Phase} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp}");

        Assert.False(host.InCombat, $"{encounterName} should have ended");
        Assert.True(
            end.Phase is GamePhase.Map or GamePhase.GameOver,
            $"{encounterName} did not resolve to a terminal state; stopped on {end.Phase}");
    }

    /// <summary>Resolve a canonical encounter of any index-0 act by its model type name.</summary>
    internal static EncounterModel ResolveEncounter(string typeName) =>
        ModelDb.ActsByIndex[0]
            .SelectMany(a => a.AllEncounters)
            .First(e => e.GetType().Name == typeName);

    private static IEnumerable<object[]> Cases(params string[] names) =>
        names.Select(n => new object[] { n });
}
