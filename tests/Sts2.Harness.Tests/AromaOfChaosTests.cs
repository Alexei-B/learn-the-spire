using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Regression guard for events whose option text is absent from our (empty) localization tables.
/// <c>EventModel.GetOptionTitle</c>/<c>GetOptionDescription</c> use <c>LocString.GetIfExists</c>,
/// which returns null for a missing key; <c>EventOption.AddLocVars</c> then dereferences the null
/// description in <c>CharacterModel.AddDetailsTo</c> and NREs, silently faulting event init so the
/// room never produced options (a forward run timed out entering it). The harness now degrades those
/// lookups to a key-named LocString, so the options generate. AromaOfChaos (a regular act-1 event
/// that builds its options from text keys) is the concrete case that used to break.
/// </summary>
public sealed class AromaOfChaosTests
{
    private readonly ITestOutputHelper _out;

    public AromaOfChaosTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task AromaOfChaos_GeneratesOptions_AndResolvesToTheMap()
    {
        var t = Task.Run(Run);
        await t.WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        EnterEvent(host, ModelDb.Event<AromaOfChaos>());

        // Option generation no longer NREs: the event surfaces both of its text-key options.
        GameState atEvent = host.GetState();
        _out.WriteLine($"phase={atEvent.Phase} eventId={atEvent.Event?.EventId} options={atEvent.Event?.Options.Count}");
        Assert.Equal(GamePhase.Event, atEvent.Phase);
        Assert.Equal("AROMA_OF_CHAOS", atEvent.Event!.EventId);
        Assert.Equal(2, atEvent.Event.Options.Count);

        var options = host.ListOptions();
        Assert.All(options, o => Assert.Equal(OptionKind.ChooseEventOption, o.Kind));
        GameOption maintainControl = options.First(o =>
            o.Description.Contains("MAINTAIN_CONTROL", StringComparison.Ordinal));

        // MAINTAIN_CONTROL upgrades a chosen deck card: the effect runs and raises a card choice.
        host.Apply(maintainControl);
        GameState atChoice = host.GetState();
        _out.WriteLine($"after choosing: phase={atChoice.Phase}");
        Assert.Equal(GamePhase.Choice, atChoice.Phase);
        var selects = host.ListOptions();
        Assert.All(selects, o => Assert.Equal(OptionKind.SelectCards, o.Kind));

        // Resolve the upgrade pick; the event then finishes and the player can move on the map.
        host.Apply(selects.First());
        GameState after = host.GetState();
        _out.WriteLine($"after resolving choice: phase={after.Phase}");
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);
    }

    /// <summary>
    /// Enter a specific (canonical) event room directly and pump until its options are generated.
    /// Mirrors the harness's own room-entry wait (<c>GameHost.WaitForEventReady</c>), which we can't
    /// reach for an arbitrary event without a seed that routes the map to it.
    /// </summary>
    private static void EnterEvent(GameHost host, EventModel canonicalEvent)
    {
        Pump(RunManager.Instance.EnterRoom(new EventRoom(canonicalEvent)));
        var sw = Stopwatch.StartNew();
        while (true)
        {
            Pump(RunManager.Instance.ActionExecutor.FinishedExecutingActions());
            EventModel? ev = RunManager.Instance.EventSynchronizer.GetLocalEvent();
            if (ev is not null && (ev.IsFinished || ev.CurrentOptions.Count > 0))
            {
                return;
            }
            if (sw.ElapsedMilliseconds > 10000)
            {
                throw new TimeoutException("Timed out waiting for the event room to initialize.");
            }
            Thread.Sleep(5);
        }
    }

    private static void Pump(Task task) => task.GetAwaiter().GetResult();
    private static T Pump<T>(Task<T> task) => task.GetAwaiter().GetResult();
}
