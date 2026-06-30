using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Shared (vote-based) events: the local player drives them through the same option API, voting on
/// an option which — with a single local voter — resolves immediately. These regress a content/shim
/// gap the property fuzzer surfaced: the shared <c>JungleMazeAdventure</c> event's "Join Forces"
/// option deref'd the null <c>NDebugAudioManager.Instance</c> for a cosmetic SFX, NRE-ing its
/// fire-and-forget effect task *before* the gold payout. With that singleton made inert (and its
/// Play/Stop no-op'd), the option now runs to completion.
/// </summary>
public sealed class SharedEventTests
{
    private readonly ITestOutputHelper _out;

    public SharedEventTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task JungleMazeAdventure_JoinForces_CompletesAndPaysOut()
    {
        await Task.Run(RunJoinForces).WaitAsync(TimeSpan.FromSeconds(60));
    }

    [Fact]
    public async Task DenseVegetation_Rest_HealsAndAdvances()
    {
        await Task.Run(RunDenseVegetationRest).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void RunDenseVegetationRest()
    {
        GameHost host = TestNav.StartOnMap("DENSE");

        // The "Rest" option heals the local player, then runs a cosmetic block that — for the local
        // player headless — used to NRE on the null debug-audio singleton and NGame.Instance.ScreenRumble
        // *after* the heal but before finishing, leaving the event mid-resolution. Lower HP so the heal
        // is observable, then choose Rest.
        TestNav.SetHp(host, maxHp: 80, currentHp: 20);

        EventModel ev = Act1EventsTests.ResolveEvent("DenseVegetation");
        Assert.True(ev.IsShared, "DenseVegetation should be a shared (vote-based) event");
        host.EnterEventDebug(ev);

        int hpBefore = host.Run.Players[0].Creature.CurrentHp;
        GameOption rest = host.ListOptions()
            .First(o => o.Kind == OptionKind.ChooseEventOption
                        && o.Description.Contains("REST", StringComparison.Ordinal));
        host.Apply(rest);

        int hpAfter = host.Run.Players[0].Creature.CurrentHp;
        _out.WriteLine($"hp {hpBefore} -> {hpAfter}");

        // The whole option ran: the heal applied (HP rose) and the event advanced to its Fight follow-up.
        Assert.True(hpAfter > hpBefore, $"Rest should have healed (was {hpBefore}, now {hpAfter})");
        Assert.Contains(host.ListOptions(), o =>
            o.Kind == OptionKind.ChooseEventOption && o.Description.Contains("FIGHT", StringComparison.Ordinal));
    }

    private void RunJoinForces()
    {
        GameHost host = TestNav.StartOnMap("JUNGLE");

        EventModel ev = Act1EventsTests.ResolveEvent("JungleMazeAdventure");
        Assert.True(ev.IsShared, "JungleMazeAdventure should be a shared (vote-based) event");
        host.EnterEventDebug(ev);
        Assert.Equal(GamePhase.Event, host.GetState().Phase);

        int goldBefore = host.Run.Players[0].Gold;

        // Choose "Join Forces" (SafetyInNumbers) — the option that used to NRE on the null debug-audio
        // singleton before paying out its gold.
        GameOption joinForces = host.ListOptions()
            .First(o => o.Kind == OptionKind.ChooseEventOption
                        && o.EventOptionRelicId is null
                        && o.Description.Contains("JOIN_FORCES", StringComparison.Ordinal));
        host.Apply(joinForces);

        int goldAfter = host.Run.Players[0].Gold;
        _out.WriteLine($"gold {goldBefore} -> {goldAfter}");

        // The effect ran to completion: gold was granted (JoinForcesGold = 50 ± 15, always positive)
        // and the event finished, so it is no longer actionable (the player leaves by moving on).
        Assert.True(goldAfter > goldBefore,
            $"Join Forces should have paid out gold (was {goldBefore}, now {goldAfter})");
        Assert.False(host.GetState().Phase == GamePhase.Event && host.ListOptions()
            .Any(o => o.Kind == OptionKind.ChooseEventOption),
            "the event should be finished (no further choices) after Join Forces");
    }
}
