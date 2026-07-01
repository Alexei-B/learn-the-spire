using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Enumerates every act-2 (<c>Hive</c>) fight and drives it to resolution through the public option
/// API, mirroring <see cref="Act1FightsTests"/>: a fresh run, the opening Neow event resolved, the
/// player buffed to a huge HP pool, then the encounter entered directly via
/// <see cref="GameHost.EnterEncounterDebug"/> and played out (block, focus-fire, end turn) until it
/// reaches a terminal state — back on the map (won) or game over (died). A fight "works and
/// resolves" if the harness drives it from start to post-combat rewards without throwing or hanging.
/// </summary>
public sealed class Act2FightsTests
{
    private readonly ITestOutputHelper _out;

    public Act2FightsTests(ITestOutputHelper output) => _out = output;

    // The discrete list of Hive (default act 2) encounters: normals/weaks, elites, bosses.
    public static IEnumerable<object[]> HiveFights => Cases(
        // Normal + weak monster encounters
        "BowlbugsNormal", "BowlbugsWeak", "ChompersNormal", "ExoskeletonsNormal", "ExoskeletonsWeak",
        "HunterKillerNormal", "LouseProgenitorNormal", "MytesNormal", "OvicopterNormal",
        "SlumberingBeetleNormal", "SpinyToadNormal", "TheObscuraNormal", "ThievingHopperWeak",
        "TunnelerWeak",
        // Elites
        "DecimillipedeElite", "EntomancerElite", "InfestedPrismsElite",
        // Bosses. KnowledgeDemon raises a player card choice during its own (enemy) turn
        // (ChooseCurse → CardSelectCmd.FromChooseACardScreen); the harness now surfaces that as a
        // GamePhase.Choice from the enemy-turn wait and resumes the turn once it is resolved.
        "KaiserCrabBoss", "KnowledgeDemonBoss", "TheInsatiableBoss");

    [Theory]
    [MemberData(nameof(HiveFights))]
    public async Task HiveFight_Resolves(string encounterName)
    {
        var t = Task.Run(() => RunFight(encounterName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunFight(string encounterName)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        EncounterModel encounter = ResolveEncounter(encounterName);
        _out.WriteLine($"Entering {encounterName} ({encounter.RoomType})");
        host.EnterEncounterDebug(encounter);

        Assert.True(host.InCombat, $"expected to be in combat after entering {encounterName}");

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

    /// <summary>Resolve a canonical act-2 (index-1) encounter by its model type name.</summary>
    internal static EncounterModel ResolveEncounter(string typeName) =>
        ModelDb.ActsByIndex[1]
            .SelectMany(a => a.AllEncounters)
            .First(e => e.GetType().Name == typeName);

    private static IEnumerable<object[]> Cases(params string[] names) =>
        names.Select(n => new object[] { n });
}
