using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Exercises the post-combat rewards flow (M2): winning a fight surfaces a
/// <see cref="GamePhase.Reward"/> screen whose gold/potion/relic/card rewards are taken via
/// <see cref="OptionKind.TakeReward"/> and left via <see cref="OptionKind.ProceedFromRewards"/>,
/// after which the run returns to the map.
/// </summary>
public sealed class RewardsTests
{
    private readonly ITestOutputHelper _out;

    public RewardsTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void WinningCombat_OffersRewards_TakeGoldAndCard_ThenProceedToMap()
    {
        var t = Task.Run(RunRewardsFlow);
        Assert.True(t.Wait(TimeSpan.FromSeconds(90)), "rewards flow did not finish within 90s");
        if (t.IsFaulted)
        {
            throw t.Exception!.Flatten().InnerExceptions.First();
        }
    }

    private void RunRewardsFlow()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();
        GameOption move = host.ListOptions().First(o => o.Kind == OptionKind.MoveTo);
        host.Apply(move);
        Assert.True(host.InCombat, "expected to land in combat after the first move");

        PlayUntilCombatEnds(host, maxTurns: 50);
        Assert.False(host.InCombat, "combat should have ended");

        // Winning a fight surfaces the rewards screen.
        GameState atRewards = host.GetState();
        _out.WriteLine($"phase={atRewards.Phase} rewards=[{DescribeRewards(atRewards)}]");
        Assert.Equal(GamePhase.Reward, atRewards.Phase);
        Assert.NotNull(atRewards.Rewards);
        Assert.NotEmpty(atRewards.Rewards!.Rewards);

        // The opening fight always grants gold and a card reward.
        Assert.Contains(atRewards.Rewards.Rewards, r => r.Type == RewardType.Gold);
        RewardView cardReward = Assert.Single(atRewards.Rewards.Rewards, r => r.Type == RewardType.Card);
        Assert.Equal(3, cardReward.Cards!.Count);

        int goldBefore = atRewards.Players[0].Gold;
        int deckBefore = atRewards.Players[0].Deck.Count;

        // Take the gold reward.
        var options = host.ListOptions();
        foreach (GameOption o in options)
        {
            _out.WriteLine($"{o.Kind}: {o.Description}");
        }
        GameOption takeGold = options.First(o => o.Kind == OptionKind.TakeReward && o.Description.Contains("gold"));
        host.Apply(takeGold);

        GameState afterGold = host.GetState();
        Assert.True(afterGold.Players[0].Gold > goldBefore, "gold should have increased after taking the gold reward");
        Assert.Equal(GamePhase.Reward, afterGold.Phase); // still on the rewards screen
        Assert.DoesNotContain( // the gold reward is now taken, so no take-gold option remains
            host.ListOptions(),
            o => o.Kind == OptionKind.TakeReward && o.Description.Contains("gold"));

        // Pick the first offered card from the card reward; it should be added to the deck.
        GameOption takeCard = host.ListOptions()
            .First(o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take card"));
        string chosenCard = takeCard.Card!.CardId;
        host.Apply(takeCard);

        GameState afterCard = host.GetState();
        Assert.Equal(deckBefore + 1, afterCard.Players[0].Deck.Count);
        Assert.Contains(afterCard.Players[0].Deck, c => c.CardId == chosenCard);
        _out.WriteLine($"took {chosenCard}; deck now {afterCard.Players[0].Deck.Count} cards");

        // Proceed: leaves the rewards screen (skipping any remaining rewards) back to the map.
        GameOption proceed = host.ListOptions().Single(o => o.Kind == OptionKind.ProceedFromRewards);
        host.Apply(proceed);

        GameState onMap = host.GetState();
        Assert.Null(onMap.Rewards);
        Assert.Equal(GamePhase.Map, onMap.Phase);

        // And we can pick the next room from the map.
        Assert.Contains(host.ListOptions(), o => o.Kind == OptionKind.MoveTo);
    }

    [Fact]
    public void ProceedingFromRewards_SkipsUntakenCard_LeavingDeckUnchanged()
    {
        var t = Task.Run(RunSkipCardReward);
        Assert.True(t.Wait(TimeSpan.FromSeconds(90)), "skip-card flow did not finish within 90s");
        if (t.IsFaulted)
        {
            throw t.Exception!.Flatten().InnerExceptions.First();
        }
    }

    private void RunSkipCardReward()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();
        host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.MoveTo));
        Assert.True(host.InCombat);

        PlayUntilCombatEnds(host, maxTurns: 50);
        Assert.Equal(GamePhase.Reward, host.GetState().Phase);

        int deckBefore = host.GetState().Players[0].Deck.Count;

        // Proceed straight away without picking a card: the card reward is skipped.
        GameOption proceed = host.ListOptions().Single(o => o.Kind == OptionKind.ProceedFromRewards);
        host.Apply(proceed);

        GameState onMap = host.GetState();
        Assert.Equal(GamePhase.Map, onMap.Phase);
        Assert.Null(onMap.Rewards);
        Assert.Equal(deckBefore, onMap.Players[0].Deck.Count); // no card added
    }

    private static string DescribeRewards(GameState s) =>
        s.Rewards is null ? "" : string.Join(", ", s.Rewards.Rewards.Select(r => r.Type.ToString()));

    private static void PlayUntilCombatEnds(GameHost host, int maxTurns)
    {
        int turns = 0;
        while (host.InCombat && turns < maxTurns)
        {
            CombatState combat = host.Combat!;
            Player player = combat.Players.Single();
            PlayerCombatState pcs = player.PlayerCombatState!;

            bool playedSomething = true;
            int guard = 0;
            while (playedSomething && host.InCombat && guard++ < 100)
            {
                playedSomething = false;
                foreach (CardModel card in pcs.Hand.Cards.ToList())
                {
                    if (!card.CanPlay(out _, out _))
                    {
                        continue;
                    }
                    Creature? target = null;
                    if (card.TargetType == TargetType.AnyEnemy)
                    {
                        target = combat.HittableEnemies.FirstOrDefault();
                        if (target == null)
                        {
                            continue;
                        }
                    }
                    if (host.PlayCard(card, target))
                    {
                        playedSomething = true;
                        break;
                    }
                }
            }

            if (host.InCombat)
            {
                host.EndTurn(player);
            }
            turns++;
        }
    }
}
