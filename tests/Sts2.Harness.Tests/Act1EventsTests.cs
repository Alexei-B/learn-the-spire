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
/// Enumerates every act-1 event (the discrete event list of each index-0 act, plus the shared
/// events) and drives it to resolution through the public option API, to catch events that NRE or
/// lock up. Each case starts a fresh run, resolves the opening Neow event, buffs the player (some
/// events deal damage or start combat), then enters the event directly (bypassing map navigation)
/// and resolves its options greedily until the event finishes and the player is back on the map. An
/// event "works and resolves" if the harness drives its primary path end-to-end without throwing.
///
/// Act 1 today means the two index-0 acts: <c>Overgrowth</c> (the default) and <c>Underdocks</c>.
/// </summary>
public sealed class Act1EventsTests
{
    private readonly ITestOutputHelper _out;

    public Act1EventsTests(ITestOutputHelper output) => _out = output;

    // Overgrowth (default act 1) events.
    public static IEnumerable<object[]> OvergrowthEvents => Cases(
        "AromaOfChaos", "ByrdonisNest", "DenseVegetation", "JungleMazeAdventure", "LuminousChoir",
        "MorphicGrove", "SapphireSeed", "SunkenStatue", "TabletOfTruth", "UnrestSite", "Wellspring",
        "WhisperingHollow", "WoodCarvings");

    // Underdocks (the alternate index-0 act) events. SunkenStatue is shared with Overgrowth and
    // already covered above, so it is omitted here to avoid a duplicate case. PunchOff is a
    // known UI-seam gap (see KnownUiGapEvents).
    public static IEnumerable<object[]> UnderdocksEvents => Cases(
        "AbyssalBaths", "DrowningBeacon", "EndlessConveyor", "SpiralingWhirlpool",
        "SunkenTreasury", "DoorsOfLightAndDark", "TrashHeap", "WaterloggedScriptorium");

    // Events shared across all acts (ModelDb.AllSharedEvents) — also reachable in act 1.
    public static IEnumerable<object[]> SharedEvents => Cases(
        "BrainLeech", "CrystalSphere", "DollRoom", "FakeMerchant", "PotionCourier", "RanwidTheElder",
        "RelicTrader", "RoomFullOfCheese", "SelfHelpBook", "SlipperyBridge", "StoneOfAllTime",
        "Symbiote", "TeaMaster", "TheFutureOfPotions", "TheLegendsWereTrue", "ThisOrThat",
        "WarHistorianRepy", "WelcomeToWongos");

    // Events whose primary path depends on an interactive UI seam the headless harness does not yet
    // model, so they cannot resolve through the option API today:
    //  - PunchOff: its "Nab" option calls NGame.Instance.ScreenShakeTrauma unguarded; NGame.Instance
    //    is null headless and the call is a callvirt (NREs at the call site, before any patch could
    //    intercept), and forcing NGame.Instance non-null would break the many NGame.Instance?.…
    //    guards other paths rely on.
    // Tracked as a follow-up (an inert NGame screen-shake seam).
    public static IEnumerable<object[]> KnownUiGapEvents => Cases("PunchOff");

    [Theory]
    [MemberData(nameof(OvergrowthEvents))]
    public async Task OvergrowthEvent_Resolves(string eventName)
    {
        var t = Task.Run(() => RunEvent(eventName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    [Theory]
    [MemberData(nameof(UnderdocksEvents))]
    public async Task UnderdocksEvent_Resolves(string eventName)
    {
        var t = Task.Run(() => RunEvent(eventName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    [Theory]
    [MemberData(nameof(SharedEvents))]
    public async Task SharedEvent_Resolves(string eventName)
    {
        var t = Task.Run(() => RunEvent(eventName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    // Enumerated but skipped: these events need an interactive UI seam the headless harness does not
    // model yet (see KnownUiGapEvents). Kept here so the gap stays visible in the test list.
    [Theory(Skip = "Needs an inert UI seam (screen-shake / event minigame); tracked as M3 follow-up.")]
    [MemberData(nameof(KnownUiGapEvents))]
    public async Task UiGapEvent_Resolves(string eventName)
    {
        var t = Task.Run(() => RunEvent(eventName));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunEvent(string eventName)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Some events deal damage or transition into combat; buff so a forced fight is survivable.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        EventModel ev = ResolveEvent(eventName);
        _out.WriteLine($"Entering event {eventName}");
        host.EnterEventDebug(ev);

        // Resolve the event greedily (and any combat/rewards it spawns) until it reaches a terminal
        // state — back on the map (finished) or game over (a spawned combat killed the player). An
        // event "resolves" when the harness drives its primary path to that terminal state without
        // throwing or hanging.
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

    /// <summary>Resolve a canonical event of any index-0 act (or a shared event) by its type name.</summary>
    internal static EventModel ResolveEvent(string typeName) =>
        ModelDb.ActsByIndex[0]
            .SelectMany(a => a.AllEvents)
            .Concat(ModelDb.AllSharedEvents)
            .First(e => e.GetType().Name == typeName);

    private static IEnumerable<object[]> Cases(params string[] names) =>
        names.Select(n => new object[] { n });
}
