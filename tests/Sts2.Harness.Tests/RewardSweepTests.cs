using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rewards;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Property-style sweep over the reward screen: across several seeds it plays a run forward (buffed
/// to survive), and at every <see cref="GamePhase.Reward"/> screen reached it exercises the option
/// kinds — reroll a card reward when available, take gold, take a card — asserting invariants
/// (gold only grows, the deck grows by exactly one per card taken, HP is untouched, a reroll leaves
/// the deck unchanged and the screen open) before proceeding (which skips any untaken rewards). The
/// guard is that every reward screen on a real forward run drives through all its options without the
/// harness throwing and without corrupting run state. Ties together the M4-deferred reward work.
/// </summary>
public sealed class RewardSweepTests
{
    private readonly ITestOutputHelper _out;

    public RewardSweepTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData("TESTSEED")]
    [InlineData("REWARDS1")]
    [InlineData("REWARDS2")]
    public async Task ForwardRun_EveryRewardScreen_DrivesAllOptions(string seed)
    {
        await Task.Run(() => RunRewardSweep(seed)).WaitAsync(TimeSpan.FromSeconds(180));
    }

    private void RunRewardSweep(string seed)
    {
        GameHost host = TestNav.StartOnMap(seed);
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        const int maxRewardScreens = 8;
        int handled = 0;
        while (handled < maxRewardScreens)
        {
            GameState s = AutoPlayer.Advance(
                host,
                stop: st => st.Phase == GamePhase.Reward || st.Phase == GamePhase.GameOver,
                maxSteps: 3000,
                log: _out);

            if (s.Phase == GamePhase.GameOver)
            {
                break;
            }

            ExerciseRewardScreen(host);
            handled++;
        }

        _out.WriteLine($"seed {seed}: exercised {handled} reward screen(s)");
        Assert.True(handled > 0, $"seed {seed} reached no reward screens");
    }

    /// <summary>Drive one reward screen through reroll/take/proceed, asserting invariants.</summary>
    private void ExerciseRewardScreen(GameHost host)
    {
        GameState before = host.GetState();
        Assert.Equal(GamePhase.Reward, before.Phase);
        int goldBefore = before.Players[0].Gold;
        int deckBefore = before.Players[0].Deck.Count;
        int hpBefore = before.Players[0].CurrentHp;

        // A card reward may offer a REROLL alternative (Driftwood); rerolling re-rolls the offered
        // cards in place — the deck is unchanged and the screen stays open.
        GameOption? reroll = host.ListOptions().FirstOrDefault(o =>
            o.Kind == OptionKind.TakeCardRewardAlternative
            && string.Equals(o.CardRewardAlternativeId, "REROLL", StringComparison.Ordinal));
        if (reroll is not null)
        {
            host.Apply(reroll);
            GameState afterReroll = host.GetState();
            Assert.Equal(GamePhase.Reward, afterReroll.Phase);
            Assert.Equal(deckBefore, afterReroll.Players[0].Deck.Count);
        }

        // Take the gold reward if present: gold strictly increases, the option then disappears.
        if (host.GetState().Rewards!.Rewards.Any(r => r.Type == RewardType.Gold && !r.Taken))
        {
            GameOption takeGold = host.ListOptions()
                .First(o => o.Kind == OptionKind.TakeReward && o.Description.Contains("gold"));
            host.Apply(takeGold);
            Assert.True(host.GetState().Players[0].Gold > goldBefore, "taking gold should increase gold");
        }

        // Take the first card of a card reward if present: the deck grows by exactly one.
        GameOption? takeCard = host.ListOptions().FirstOrDefault(o =>
            o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take card", StringComparison.Ordinal));
        if (takeCard is not null)
        {
            int deck = host.GetState().Players[0].Deck.Count;
            host.Apply(takeCard);
            Assert.Equal(deck + 1, host.GetState().Players[0].Deck.Count);
        }

        // Rewards never change HP; the only mutations are gold/deck/relic/potion.
        Assert.Equal(hpBefore, host.GetState().Players[0].CurrentHp);

        // Proceed: skips any untaken rewards (relics/potions) and leaves the reward screen.
        GameOption proceed = host.ListOptions().Single(o => o.Kind == OptionKind.ProceedFromRewards);
        host.Apply(proceed);
        Assert.NotEqual(GamePhase.Reward, host.GetState().Phase);
    }
}
