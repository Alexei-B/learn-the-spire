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
/// Enumerates every act-3 (<c>Glory</c>) fight and drives it to resolution through the public option
/// API, mirroring <see cref="Act1FightsTests"/> / <see cref="Act2FightsTests"/>: a fresh run, the
/// opening Neow event resolved, the player buffed to a huge HP pool, then the encounter entered
/// directly via <see cref="GameHost.EnterEncounterDebug"/> and played out until it reaches a terminal
/// state — back on the map (won) or game over (died). A fight "works and resolves" if the harness
/// drives it from start to post-combat rewards without throwing or hanging.
/// </summary>
public sealed class Act3FightsTests
{
    private readonly ITestOutputHelper _out;

    public Act3FightsTests(ITestOutputHelper output) => _out = output;

    // The discrete list of Glory (default act 3) encounters: normals/weaks, elites, bosses.
    public static IEnumerable<object[]> GloryFights => Cases(
        // Normal + weak monster encounters
        "AxebotsNormal", "ConstructMenagerieNormal", "DevotedSculptorWeak", "FabricatorNormal",
        "FrogKnightNormal", "GlobeHeadNormal", "OwlMagistrateNormal", "ScrollsOfBitingNormal",
        "ScrollsOfBitingWeak", "SlimedBerserkerNormal", "TheLostAndForgottenNormal",
        "TurretOperatorWeak",
        // Elites
        "KnightsElite", "MechaKnightElite", "SoulNexusElite",
        // Bosses
        "AeonglassBoss", "QueenBoss", "TestSubjectBoss");

    [Theory]
    [MemberData(nameof(GloryFights))]
    public async Task GloryFight_Resolves(string encounterName)
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

    /// <summary>Resolve a canonical act-3 (index-2) encounter by its model type name.</summary>
    internal static EncounterModel ResolveEncounter(string typeName) =>
        ModelDb.ActsByIndex[2]
            .SelectMany(a => a.AllEncounters)
            .First(e => e.GetType().Name == typeName);

    private static IEnumerable<object[]> Cases(params string[] names) =>
        names.Select(n => new object[] { n });
}
