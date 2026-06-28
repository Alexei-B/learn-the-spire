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

    // Glory (default act 3) events.
    public static IEnumerable<object[]> GloryEvents => Cases(
        "BattlewornDummy", "GraveOfTheForgotten", "HungryForMushrooms", "Reflections",
        "RoundTeaParty", "TinkerTime");

    // Trial is intentionally omitted: its first option (Accept) drives the event through the event
    // room's portrait UI — NEventRoom.Instance.Layout.RemoveNodesOnPortrait()/SetPortrait()/
    // AddVfxAnchoredToPortrait() plus a scene instantiate — all unguarded on the null headless
    // NEventRoom singleton, so Accept NREs before it builds the (mechanically meaningful) verdict
    // sub-options and the greedy driver loops. The portrait coupling cascades through NEventLayout,
    // so a faithful headless stand-in is more event-UI plumbing than this milestone takes on; the
    // verdict options themselves (curses/relics/rewards/card-selects) are ordinary and would resolve.

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
