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
/// Enumerates every act-2 (<c>Hive</c>) event and drives it to resolution through the public option
/// API, mirroring <see cref="Act1EventsTests"/>: a fresh run, the opening Neow event resolved, the
/// player buffed (some events deal damage or start combat), then the event entered directly via
/// <see cref="GameHost.EnterEventDebug"/> and its options resolved greedily until it finishes and the
/// player is back on the map (or a spawned combat ends the run). Shared events are already covered by
/// <see cref="Act1EventsTests"/>, so only Hive's own events are listed here.
/// </summary>
public sealed class Act2EventsTests
{
    private readonly ITestOutputHelper _out;

    public Act2EventsTests(ITestOutputHelper output) => _out = output;

    // Hive (default act 2) events.
    public static IEnumerable<object[]> HiveEvents => Cases(
        "Amalgamator", "Bugslayer", "ColorfulPhilosophers", "ColossalFlower", "FieldOfManSizedHoles",
        "InfestedAutomaton", "LostWisp", "SpiritGrafter", "TheLanternKey", "ZenWeaver");

    [Theory]
    [MemberData(nameof(HiveEvents))]
    public async Task HiveEvent_Resolves(string eventName)
    {
        var t = Task.Run(() => RunEvent(eventName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunEvent(string eventName)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        EventModel ev = ResolveEvent(eventName);
        _out.WriteLine($"Entering event {eventName}");
        host.EnterEventDebug(ev);

        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
            maxSteps: 3000,
            log: _out);

        _out.WriteLine($"{eventName} ended: phase={end.Phase} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp}");

        Assert.True(
            end.Phase is GamePhase.Map or GamePhase.GameOver,
            $"{eventName} did not resolve to a terminal state; stopped on {end.Phase}");
    }

    /// <summary>Resolve a canonical act-2 (index-1) event by its type name.</summary>
    internal static EventModel ResolveEvent(string typeName) =>
        ModelDb.ActsByIndex[1]
            .SelectMany(a => a.AllEvents)
            .First(e => e.GetType().Name == typeName);

    private static IEnumerable<object[]> Cases(params string[] names) =>
        names.Select(n => new object[] { n });
}
