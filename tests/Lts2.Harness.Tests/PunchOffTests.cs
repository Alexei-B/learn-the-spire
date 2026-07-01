using System.Linq;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises the PunchOff event's "Nab" option, whose effect calls an unguarded
/// <c>NGame.Instance.ScreenShakeTrauma</c> — a cosmetic screen-shake on a UI singleton that is null
/// headless. A Harmony transpiler strips just that call from the option's IL so the rest of the
/// effect (add the Injury curse, then grant a relic via the custom-reward screen, then finish) runs
/// instead of NRE'ing at the call site. This guards that the option executes past that point and
/// pays out, not merely that the event doesn't crash.
/// </summary>
public sealed class PunchOffTests
{
    private readonly ITestOutputHelper _out;

    public PunchOffTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void NabOption_AddsCurseAndGrantsRelic_ThenReturnsToMap()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        int relicsBefore = host.GetState().Players[0].Relics.Count;
        int deckBefore = host.GetState().Players[0].Deck.Count;

        host.EnterEventDebug(Act1EventsTests.ResolveEvent("PunchOff"));

        GameState atEvent = host.GetState();
        Assert.Equal(GamePhase.Event, atEvent.Phase);
        int nabIndex = atEvent.Event!.Options.Single(o => o.TextKey.Contains("NAB")).Index;
        host.Apply(host.ListOptions().Single(o =>
            o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == nabIndex));

        // The effect ran past the (stripped) screen-shake call and offered its relic reward — so we
        // land on the reward screen rather than having NRE'd partway through.
        GameState atReward = host.GetState();
        _out.WriteLine($"after Nab: phase={atReward.Phase} rewards=[{string.Join(",", atReward.Rewards?.Rewards.Select(r => r.Type.ToString()) ?? Enumerable.Empty<string>())}]");
        Assert.Equal(GamePhase.Reward, atReward.Phase);

        // Take the relic and proceed back to the map.
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
            maxSteps: 100,
            log: _out);

        PlayerState player = end.Players[0];
        _out.WriteLine($"ended phase={end.Phase} relics={player.Relics.Count} deck={player.Deck.Count}");

        Assert.Equal(GamePhase.Map, end.Phase);
        Assert.Equal(relicsBefore + 1, player.Relics.Count); // Nab's relic reward was granted
        Assert.True(player.Deck.Count > deckBefore, "Nab adds an Injury curse to the deck");
    }
}
