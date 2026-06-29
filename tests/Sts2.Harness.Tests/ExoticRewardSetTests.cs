using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Covers the "exotic" relics whose on-obtain effect spawns a <see cref="RewardsCmd.OfferCustom"/>
/// reward set, beyond Kaleidoscope's two bonus card rewards. Each of these routes through the
/// harness's custom-reward gate (<c>OnCustomRewardsOffered</c> / <c>RewardsSet.testSelector</c>); the
/// relic sweep already drives them to completion, but these tests assert the *shape* of each set —
/// card / potion / relic / mixed — surfaces correctly through the public option API and that taking
/// rewards has the expected effect. This is the test-and-cover task from the M4-deferred roadmap.
/// </summary>
public sealed class ExoticRewardSetTests
{
    private readonly ITestOutputHelper _out;

    public ExoticRewardSetTests(ITestOutputHelper output) => _out = output;

    /// <summary>CallingBell offers a relic-pick set of three relics (and adds a curse).</summary>
    [Fact]
    public async Task CallingBell_OffersThreeRelicPicks_AndAddsCurse()
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.StartOnMap("TESTSEED");
            int deckBefore = host.GetState().Players[0].Deck.Count;

            host.ObtainRelicDebug(Relic("CALLING_BELL"));

            GameState atRewards = host.GetState();
            Assert.Equal(GamePhase.Reward, atRewards.Phase);
            var relicRewards = atRewards.Rewards!.Rewards.Where(r => r.Type == RewardType.Relic).ToList();
            Assert.Equal(3, relicRewards.Count);
            _out.WriteLine($"CallingBell relic picks: {string.Join(", ", relicRewards.Select(r => r.RelicId))}");

            // CallingBell adds CurseOfTheBell to the deck as part of its obtain effect.
            Assert.True(
                host.GetState().Players[0].Deck.Count > deckBefore,
                "CallingBell should have added a curse to the deck");

            // Take one of the three relics; one of the offered relics lands on the player.
            var offered = relicRewards.Select(r => r.RelicId).ToHashSet();
            int relicCountBefore = host.GetState().Players[0].Relics.Count;
            GameOption takeRelic = host.ListOptions()
                .First(o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take relic", StringComparison.Ordinal));
            host.Apply(takeRelic);
            var relicsAfter = host.GetState().Players[0].Relics;
            Assert.Equal(relicCountBefore + 1, relicsAfter.Count);
            Assert.Contains(relicsAfter, r => offered.Contains(r));

            ProceedToMap(host);
        }).WaitAsync(TimeSpan.FromSeconds(90));
    }

    /// <summary>Cauldron offers a potion set; taking them fills the belt (extra are no-ops).</summary>
    [Fact]
    public async Task Cauldron_OffersPotionSet_TakingFillsBelt()
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.StartOnMap("TESTSEED");

            host.ObtainRelicDebug(Relic("CAULDRON"));

            GameState atRewards = host.GetState();
            Assert.Equal(GamePhase.Reward, atRewards.Phase);
            int potionRewards = atRewards.Rewards!.Rewards.Count(r => r.Type == RewardType.Potion);
            Assert.Equal(5, potionRewards); // Cauldron's CanonicalVars: 5 potions

            int beltSize = atRewards.Players[0].Potions.Count;

            // Take potions until the belt is full; the remaining offered potions stay untaken.
            GameOption? take;
            while ((take = host.ListOptions().FirstOrDefault(o =>
                       o.Kind == OptionKind.TakeReward
                       && o.Description.StartsWith("Take potion", StringComparison.Ordinal)
                       && host.GetState().Players[0].Potions.Any(p => p is null))) is not null)
            {
                host.Apply(take);
            }

            int held = host.GetState().Players[0].Potions.Count(p => p is not null);
            Assert.True(held > 0 && held <= beltSize, $"expected 1..{beltSize} potions held, got {held}");
            _out.WriteLine($"Cauldron filled belt: {held}/{beltSize}");

            ProceedToMap(host);
        }).WaitAsync(TimeSpan.FromSeconds(90));
    }

    /// <summary>LostCoffer offers a mixed card+potion set.</summary>
    [Fact]
    public async Task LostCoffer_OffersCardAndPotion()
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.StartOnMap("TESTSEED");
            int deckBefore = host.GetState().Players[0].Deck.Count;

            host.ObtainRelicDebug(Relic("LOST_COFFER"));

            GameState atRewards = host.GetState();
            Assert.Equal(GamePhase.Reward, atRewards.Phase);
            Assert.Contains(atRewards.Rewards!.Rewards, r => r.Type == RewardType.Card);
            Assert.Contains(atRewards.Rewards.Rewards, r => r.Type == RewardType.Potion);

            // Take the card reward's first card; the deck grows by one.
            GameOption takeCard = host.ListOptions()
                .First(o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take card", StringComparison.Ordinal));
            host.Apply(takeCard);
            Assert.Equal(deckBefore + 1, host.GetState().Players[0].Deck.Count);

            ProceedToMap(host);
        }).WaitAsync(TimeSpan.FromSeconds(90));
    }

    /// <summary>SmallCapsule offers a single relic pick.</summary>
    [Fact]
    public async Task SmallCapsule_OffersOneRelic()
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.StartOnMap("TESTSEED");

            host.ObtainRelicDebug(Relic("SMALL_CAPSULE"));

            GameState atRewards = host.GetState();
            Assert.Equal(GamePhase.Reward, atRewards.Phase);
            RewardView relicReward = Assert.Single(atRewards.Rewards!.Rewards, r => r.Type == RewardType.Relic);
            string relicId = relicReward.RelicId!;
            Assert.NotNull(relicId);

            GameOption takeRelic = host.ListOptions()
                .First(o => o.Kind == OptionKind.TakeReward && o.Description.StartsWith("Take relic", StringComparison.Ordinal));
            host.Apply(takeRelic);
            Assert.Contains(relicId, host.GetState().Players[0].Relics);

            ProceedToMap(host);
        }).WaitAsync(TimeSpan.FromSeconds(90));
    }

    /// <summary>
    /// The remaining card-set relics (Orrery, GlassEye) and the wax-relic set (ToyBox) each surface a
    /// custom reward set with the expected top-level reward type, and proceed-skip returns to the map.
    /// </summary>
    [Theory]
    [InlineData("ORRERY", RewardType.Card, 5)]
    [InlineData("GLASS_EYE", RewardType.Card, 5)]
    [InlineData("TOY_BOX", RewardType.Relic, 4)]
    public async Task CardOrRelicSet_Surfaces_AndProceeds(string relicId, RewardType type, int count)
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.StartOnMap("TESTSEED");

            host.ObtainRelicDebug(Relic(relicId));

            GameState atRewards = host.GetState();
            Assert.Equal(GamePhase.Reward, atRewards.Phase);
            Assert.Equal(count, atRewards.Rewards!.Rewards.Count(r => r.Type == type));

            // Skip everything via proceed; the run returns to the map with the obtain effect resumed.
            ProceedToMap(host);
        }).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private static RelicModel Relic(string id) =>
        ModelDb.AllRelics.First(r => r.Id.Entry == id || r.GetType().Name == id);

    private static void ProceedToMap(GameHost host)
    {
        GameOption proceed = host.ListOptions().Single(o => o.Kind == OptionKind.ProceedFromRewards);
        host.Apply(proceed);
        Assert.Equal(GamePhase.Map, host.GetState().Phase);
    }
}
