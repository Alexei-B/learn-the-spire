using System.Linq;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using Sts2.Harness;
using Xunit;

namespace Sts2.Harness.Tests;

/// <summary>
/// Shared navigation helpers for tests. With all epochs unlocked, every run opens on the Neow
/// ancient event, so reaching the map (or the first combat) means resolving that event first.
/// </summary>
internal static class TestNav
{
    /// <summary>
    /// Start a run and advance to the map: enter the first room (the Neow ancient event) and
    /// take its first option, which finishes the event and leaves the player on the map.
    /// </summary>
    public static GameHost StartOnMap(string seed = "TESTSEED")
    {
        GameHost host = GameHost.StartNewRun(seed);
        host.EnterFirstRoom();
        ResolveOpeningAncient(host);
        return host;
    }

    /// <summary>
    /// If the run is sitting on the opening Neow ancient event, take a benign blessing so the run
    /// proceeds to the map. A no-op if not currently in an event.
    ///
    /// Blessings whose relic has an upon-pickup effect (e.g. Kaleidoscope, which spawns two bonus
    /// card rewards) are skipped, so combat/reward tests start from the normal starting deck rather
    /// than one padded by side-effect rewards.
    /// </summary>
    public static void ResolveOpeningAncient(GameHost host)
    {
        if (host.GetState().Phase != GamePhase.Event)
        {
            return;
        }

        EventModel ev = RunManager.Instance.EventSynchronizer.GetLocalEvent();
        int chosenIndex = -1;
        int fallbackIndex = -1;
        for (int i = 0; i < ev.CurrentOptions.Count; i++)
        {
            MegaCrit.Sts2.Core.Events.EventOption opt = ev.CurrentOptions[i];
            if (opt.IsLocked || opt.IsProceed)
            {
                continue;
            }
            if (fallbackIndex < 0)
            {
                fallbackIndex = i;
            }
            if (opt.Relic is { HasUponPickupEffect: true })
            {
                continue;
            }
            chosenIndex = i;
            break;
        }
        if (chosenIndex < 0)
        {
            chosenIndex = fallbackIndex;
        }

        GameOption pick = host.ListOptions().First(o => o.EventOptionIndex == chosenIndex);
        host.Apply(pick);
    }

    /// <summary>
    /// Start a run, resolve the opening ancient event, and move into the first reachable room,
    /// which on the standard seeds is the first combat.
    /// </summary>
    public static GameHost MoveIntoFirstCombat(string seed = "TESTSEED")
    {
        GameHost host = StartOnMap(seed);
        GameOption move = host.ListOptions().First(o => o.Kind == OptionKind.MoveTo);
        host.Apply(move);
        Assert.True(host.InCombat, "expected to land in combat after the first move");
        return host;
    }
}
