using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rewards;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises card-reward *alternatives* (M4 polish): the non-card choices on a post-combat card
/// reward. A terminal alternative (Pael's Wing's <c>SACRIFICE</c>) completes the reward without
/// adding a card; a <c>REROLL</c> (Driftwood) re-rolls the offered cards in place and is single-use.
/// Both surface as <see cref="OptionKind.TakeCardRewardAlternative"/> options and resolve through the
/// same selection seam as taking a card.
/// </summary>
public sealed class RewardAlternativesTests
{
    private readonly ITestOutputHelper _out;

    public RewardAlternativesTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task PaelsWing_OffersSacrifice_WhichCompletesCardRewardWithoutAddingACard()
    {
        await Task.Run(RunSacrifice).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunSacrifice()
    {
        GameState atRewards = ReachFirstCardReward("PaelsWing", out GameHost host);

        // Pael's Wing adds a SACRIFICE alternative to the card reward.
        RewardView card = Assert.Single(atRewards.Rewards!.Rewards, r => r.Type == RewardType.Card);
        Assert.Contains("SACRIFICE", card.CardAlternatives!);
        GameOption sacrifice = host.ListOptions().Single(o =>
            o.Kind == OptionKind.TakeCardRewardAlternative && o.CardRewardAlternativeId == "SACRIFICE");

        int deckBefore = atRewards.Players[0].Deck.Count;
        host.Apply(sacrifice);

        GameState after = host.GetState();
        // Sacrifice completes (consumes) the card reward without adding a card to the deck.
        Assert.Equal(deckBefore, after.Players[0].Deck.Count);
        Assert.DoesNotContain(host.ListOptions(),
            o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take card", StringComparison.Ordinal));
        Assert.DoesNotContain(host.ListOptions(),
            o => o.Kind == OptionKind.TakeCardRewardAlternative);
        // Other rewards (gold) remain, so we're still on the rewards screen and can proceed.
        Assert.Equal(GamePhase.Reward, after.Phase);
        host.Apply(host.ListOptions().Single(o => o.Kind == OptionKind.ProceedFromRewards));
        Assert.Equal(GamePhase.Map, host.GetState().Phase);
    }

    [Fact]
    public async Task Driftwood_OffersReroll_WhichReRollsTheCardsInPlace_AndIsSingleUse()
    {
        await Task.Run(RunReroll).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunReroll()
    {
        GameState atRewards = ReachFirstCardReward("Driftwood", out GameHost host);

        RewardView card = Assert.Single(atRewards.Rewards!.Rewards, r => r.Type == RewardType.Card);
        Assert.Contains("REROLL", card.CardAlternatives!);
        _out.WriteLine($"before reroll: [{string.Join(",", card.Cards!.Select(c => c.CardId))}]");

        GameOption reroll = host.ListOptions().Single(o =>
            o.Kind == OptionKind.TakeCardRewardAlternative && o.CardRewardAlternativeId == "REROLL");
        host.Apply(reroll);

        // The rewards screen stays up with a freshly-rolled set, and reroll is single-use.
        GameState after = host.GetState();
        Assert.Equal(GamePhase.Reward, after.Phase);
        RewardView card2 = Assert.Single(after.Rewards!.Rewards, r => r.Type == RewardType.Card);
        Assert.Equal(3, card2.Cards!.Count);
        Assert.DoesNotContain("REROLL", card2.CardAlternatives!);
        Assert.DoesNotContain(host.ListOptions(),
            o => o.Kind == OptionKind.TakeCardRewardAlternative && o.CardRewardAlternativeId == "REROLL");
        _out.WriteLine($"after reroll:  [{string.Join(",", card2.Cards.Select(c => c.CardId))}]");

        // A rerolled card is still takeable.
        int deckBefore = after.Players[0].Deck.Count;
        GameOption takeCard = host.ListOptions()
            .First(o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take card", StringComparison.Ordinal));
        host.Apply(takeCard);
        Assert.Equal(deckBefore + 1, host.GetState().Players[0].Deck.Count);
    }

    /// <summary>
    /// Start a run holding <paramref name="relicId"/>, win the first combat with the greedy driver,
    /// and stop on the rewards screen (before taking anything). The relic is granted before the fight
    /// so its reward-modifying hooks run when the rewards are generated.
    /// </summary>
    private GameState ReachFirstCardReward(string relicId, out GameHost host)
    {
        host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);
        TestNav.GiveRelic(host, relicId);

        host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.MoveTo));
        Assert.True(host.InCombat, "expected to land in the first combat");

        GameState atRewards = AutoPlayer.Advance(host, stop: s => s.Phase == GamePhase.Reward, log: _out);
        Assert.Equal(GamePhase.Reward, atRewards.Phase);
        return atRewards;
    }
}
