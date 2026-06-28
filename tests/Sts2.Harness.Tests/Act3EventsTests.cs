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
/// Enumerates every act-3 (<c>Glory</c>) event and drives it to resolution through the public option
/// API, mirroring <see cref="Act1EventsTests"/> / <see cref="Act2EventsTests"/>. Shared events are
/// already covered by <see cref="Act1EventsTests"/>, so only Glory's own events are listed here.
/// </summary>
public sealed class Act3EventsTests
{
    private readonly ITestOutputHelper _out;

    public Act3EventsTests(ITestOutputHelper output) => _out = output;

    // Glory (default act 3) events. Trial's Accept option drives the event-room portrait UI
    // (NEventRoom.Instance.Layout.*), which the harness neutralizes with inert NEventRoom/NEventLayout
    // stand-ins so Accept reaches its verdict sub-options (curses/relics/rewards/card-selects).
    public static IEnumerable<object[]> GloryEvents => Cases(
        "BattlewornDummy", "GraveOfTheForgotten", "HungryForMushrooms", "Reflections",
        "RoundTeaParty", "Trial", "TinkerTime");

    [Theory]
    [MemberData(nameof(GloryEvents))]
    public async Task GloryEvent_Resolves(string eventName)
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

    /// <summary>Resolve a canonical act-3 (index-2) event by its type name.</summary>
    internal static EventModel ResolveEvent(string typeName) =>
        ModelDb.ActsByIndex[2]
            .SelectMany(a => a.AllEvents)
            .First(e => e.GetType().Name == typeName);

    private static IEnumerable<object[]> Cases(params string[] names) =>
        names.Select(n => new object[] { n });
}
