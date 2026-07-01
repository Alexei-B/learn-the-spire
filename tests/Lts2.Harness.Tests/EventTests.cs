using System.Linq;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises M3 non-combat rooms — events. With all epochs unlocked, every run opens on the
/// Neow ancient event, surfaced as <see cref="GamePhase.Event"/> with one
/// <see cref="OptionKind.ChooseEventOption"/> per blessing; choosing one runs the option's
/// effect (here, obtaining a relic) and proceeds to the map. Blessings whose relic grants bonus
/// rewards (Kaleidoscope) surface those as an explicit take-or-skip reward screen.
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
    public void ChoosingABenignBlessing_GrantsTheRelic_AndProceedsToTheMap()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

        GameState atEvent = host.GetState();
        Assert.Equal(GamePhase.Event, atEvent.Phase);
        int relicsBefore = atEvent.Players[0].Relics.Count;

        // Take a blessing whose relic has no upon-pickup effect, so it cleanly grants the relic
        // and proceeds straight to the map (no bonus reward screen).
        GameOption pick = FirstBenignBlessing(host);
        string grantedRelic = pick.EventOptionRelicId!;
        _out.WriteLine($"taking blessing {pick.Description} -> relic {grantedRelic}");
        host.Apply(pick);

        GameState onMap = host.GetState();
        _out.WriteLine($"phase={onMap.Phase} relics=[{string.Join(",", onMap.Players[0].Relics)}]");

        Assert.Equal(GamePhase.Map, onMap.Phase);
        Assert.Null(onMap.Event);
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);

        Assert.Equal(relicsBefore + 1, onMap.Players[0].Relics.Count);
        Assert.Contains(grantedRelic, onMap.Players[0].Relics);
    }

    [Fact]
    public void KaleidoscopeBlessing_OffersBonusCardRewards_AsExplicitTakeOrSkip()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

        // Kaleidoscope's relic grants two bonus card rewards on pickup; they must be an explicit
        // choice, not silently auto-taken.
        GameOption kaleidoscope = host.ListOptions()
            .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionRelicId == "KALEIDOSCOPE");
        int deckBefore = host.GetState().Players[0].Deck.Count;
        host.Apply(kaleidoscope);

        // Choosing it does NOT go straight to the map: the bonus rewards surface as a reward screen.
        GameState atRewards = host.GetState();
        _out.WriteLine($"phase={atRewards.Phase} rewards=[{string.Join(", ", atRewards.Rewards?.Rewards.Select(r => $"{r.Type}({r.Cards?.Count})") ?? Enumerable.Empty<string>())}]");
        Assert.Equal(GamePhase.Reward, atRewards.Phase);
        Assert.NotNull(atRewards.Rewards);
        var cardRewards = atRewards.Rewards!.Rewards.Where(r => r.Type == RewardType.Card).ToList();
        Assert.Equal(2, cardRewards.Count); // Kaleidoscope offers two card rewards
        Assert.All(cardRewards, r => Assert.NotEmpty(r.Cards!));
        Assert.Contains("KALEIDOSCOPE", host.GetState().Players[0].Relics);

        // Take one card from the first bonus reward, then proceed — skipping the second reward.
        GameOption takeCard = host.ListOptions()
            .First(o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take card"));
        string takenCard = takeCard.Card!.CardId;
        _out.WriteLine($"taking bonus card {takenCard}");
        host.Apply(takeCard);

        // Still on the reward screen (the second bonus reward remains).
        Assert.Equal(GamePhase.Reward, host.GetState().Phase);

        GameOption proceed = host.ListOptions().Single(o => o.Kind == OptionKind.ProceedFromRewards);
        host.Apply(proceed);

        // Proceeding resumes the Neow event, which finishes and lands on the map.
        GameState onMap = host.GetState();
        _out.WriteLine($"phase={onMap.Phase} deck={onMap.Players[0].Deck.Count} (was {deckBefore})");
        Assert.Equal(GamePhase.Map, onMap.Phase);
        Assert.Null(onMap.Rewards);

        // Exactly the one bonus card we took was added (the second reward was skipped).
        Assert.Equal(deckBefore + 1, onMap.Players[0].Deck.Count);
        Assert.Contains(onMap.Players[0].Deck, c => c.CardId == takenCard);
    }

    /// <summary>The first Neow option whose relic has no upon-pickup (bonus reward) effect.</summary>
    private static GameOption FirstBenignBlessing(GameHost host)
    {
        EventModel ev = RunManager.Instance.EventSynchronizer.GetLocalEvent();
        var benign = host.ListOptions()
            .First(o => o.Kind == OptionKind.ChooseEventOption
                        && ev.CurrentOptions[o.EventOptionIndex!.Value].Relic is { HasUponPickupEffect: false });
        return benign;
    }
}
