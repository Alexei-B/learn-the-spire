using System.Linq;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Exercises M3 non-combat rooms — events. With all epochs unlocked, every run opens on the
/// Neow ancient event, surfaced as <see cref="GamePhase.Event"/> with one
/// <see cref="OptionKind.ChooseEventOption"/> per blessing; choosing one runs the option's
/// effect (here, obtaining a relic) and proceeds to the map.
/// </summary>
public sealed class EventTests
{
    private readonly ITestOutputHelper _out;

    public EventTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Run_OpensOnTheNeowAncientEvent()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

        GameState state = host.GetState();
        _out.WriteLine($"phase={state.Phase} event={state.Event?.EventId} ancient={state.Event?.IsAncient}");
        foreach (EventOptionView o in state.Event?.Options ?? System.Array.Empty<EventOptionView>())
        {
            _out.WriteLine($"  [{o.Index}] {o.TextKey} relic={o.RelicId}");
        }

        Assert.Equal(GamePhase.Event, state.Phase);
        Assert.NotNull(state.Event);
        Assert.Equal("NEOW", state.Event!.EventId);
        Assert.True(state.Event.IsAncient);
        Assert.NotEmpty(state.Event.Options);

        // Every Neow option offers a relic blessing.
        Assert.All(state.Event.Options, o => Assert.NotNull(o.RelicId));

        // The options listed match the ChooseEventOption options on offer.
        var options = host.ListOptions();
        Assert.NotEmpty(options);
        Assert.All(options, o => Assert.Equal(OptionKind.ChooseEventOption, o.Kind));
        Assert.Equal(state.Event.Options.Count, options.Count);
    }

    [Fact]
    public void ChoosingANeowBlessing_GrantsTheRelic_AndProceedsToTheMap()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

        GameState atEvent = host.GetState();
        Assert.Equal(GamePhase.Event, atEvent.Phase);
        int relicsBefore = atEvent.Players[0].Relics.Count;

        // Take the first blessing; it should grant that option's relic.
        GameOption pick = host.ListOptions().First(o => o.Kind == OptionKind.ChooseEventOption);
        string grantedRelic = pick.EventOptionRelicId!;
        _out.WriteLine($"taking blessing {pick.Description} -> relic {grantedRelic}");
        host.Apply(pick);

        GameState onMap = host.GetState();
        _out.WriteLine($"phase={onMap.Phase} relics=[{string.Join(",", onMap.Players[0].Relics)}]");

        // The event is over: we're on the map and can pick the next room.
        Assert.Equal(GamePhase.Map, onMap.Phase);
        Assert.Null(onMap.Event);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);

        // The blessing's relic was added to the deck.
        Assert.Equal(relicsBefore + 1, onMap.Players[0].Relics.Count);
        Assert.Contains(grantedRelic, onMap.Players[0].Relics);
    }
}
