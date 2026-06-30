using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// M6 — local ("fake") multiplayer: one process hosting N players on the singleplayer net service,
/// all driven by the harness. These tests cover run setup, the multi-player read model, and the
/// per-player combat turn structure (the enemy turn resolves only once *every* player has ended).
/// </summary>
public sealed class MultiplayerTests
{
    private readonly ITestOutputHelper _out;

    public MultiplayerTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void StartNewRun_WithTwoPlayers_BootsBothWithDistinctNetIdsAndDecks()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);

        GameState state = host.GetState();
        _out.WriteLine($"players={state.Players.Count}");
        foreach (PlayerState p in state.Players)
        {
            _out.WriteLine($"  netId={p.NetId} char={p.Character} hp={p.CurrentHp}/{p.MaxHp} deck={p.Deck.Count}");
        }

        Assert.Equal(2, state.Players.Count);
        Assert.Equal(new ulong[] { 1, 2 }, state.Players.Select(p => p.NetId).ToArray());
        Assert.All(state.Players, p => Assert.True(p.MaxHp > 0));
        Assert.All(state.Players, p => Assert.True(p.Deck.Count > 0));
    }

    [Fact]
    public async Task TwoPlayers_NavigateForward_EachResolvesOwnNeow_ThenThePartyVotesIntoCombat()
    {
        await Task.Run(RunTwoPlayerNavigation).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunTwoPlayerNavigation()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);
        host.EnterFirstRoom();

        // The party opens on the Neow ancient event — each player gets their *own* instance and
        // resolves it independently. While player 1 still has a choice, player 2 also has one.
        Assert.Equal(GamePhase.Event, host.GetState().Phase);

        // Resolve each player's Neow through their own per-player options. Each player has its own
        // ChooseEventOption set (the per-player event instances), surfaced by ListOptions(netId). Pick
        // a benign blessing (one whose relic has no upon-pickup side effect) so each option resolves
        // cleanly — the same clean-start choice TestNav makes for single-player.
        foreach (Player player in host.Run.Players)
        {
            int index = BenignNeowOptionIndex(player);
            GameOption blessing = host.ListOptions(player.NetId)
                .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == index);
            host.Apply(blessing);
        }

        // With both Neow events done the party is on the act map.
        GameState onMap = host.GetState();
        Assert.Equal(GamePhase.Map, onMap.Phase);

        // Move on the map by vote: player 1 votes first (no move yet — the party waits for everyone),
        // then player 2's vote completes the tally and the party moves together into the first room.
        // Both vote for the same destination so the (vote-weighted) pick is deterministic.
        Coord dest = host.ListOptions(1uL).First(o => o.Kind == OptionKind.MoveTo).Coord!.Value;

        GameOption p1Move = host.ListOptions(1uL)
            .First(o => o.Kind == OptionKind.MoveTo && o.Coord == dest);
        host.Apply(p1Move);
        Assert.False(host.InCombat, "the party should not have moved on a single vote");

        GameOption p2Move = host.ListOptions(2uL)
            .First(o => o.Kind == OptionKind.MoveTo && o.Coord == dest);
        host.Apply(p2Move);

        Assert.True(host.InCombat, "the party should have moved into combat once both players voted");
        Assert.All(host.Run.Players, p => Assert.NotNull(p.PlayerCombatState));
        _out.WriteLine($"both players in combat at floor {host.GetState().Floor}");
    }

    /// <summary>
    /// The index of a benign Neow option for the given player's own event: a non-locked, non-proceed
    /// blessing whose relic has no upon-pickup side effect (so choosing it resolves cleanly, leaving
    /// the starting deck unpadded). Falls back to the first actionable option.
    /// </summary>
    private static int BenignNeowOptionIndex(Player player)
    {
        MegaCrit.Sts2.Core.Models.EventModel ev =
            MegaCrit.Sts2.Core.Runs.RunManager.Instance.EventSynchronizer.GetEventForPlayer(player);
        int fallback = -1;
        for (int i = 0; i < ev.CurrentOptions.Count; i++)
        {
            MegaCrit.Sts2.Core.Events.EventOption opt = ev.CurrentOptions[i];
            if (opt.IsLocked || opt.IsProceed)
            {
                continue;
            }
            if (fallback < 0)
            {
                fallback = i;
            }
            if (opt.Relic is { HasUponPickupEffect: true })
            {
                continue;
            }
            return i;
        }
        return fallback;
    }

    [Fact]
    public async Task TwoPlayers_SharedEvent_EachVotes_AndSeesTheOthersIndicatedChoice()
    {
        await Task.Run(RunSharedEventVoting).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunSharedEventVoting()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);
        host.EnterFirstRoom();
        foreach (Player player in host.Run.Players) // resolve each player's own Neow
        {
            int idx = BenignNeowOptionIndex(player);
            host.Apply(host.ListOptions(player.NetId)
                .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == idx));
        }

        // Drop the party into a shared (vote-based) event. Both players get their own instance and
        // vote on a single shared option.
        EventModel ev = Act1EventsTests.ResolveEvent("DenseVegetation");
        Assert.True(ev.IsShared, "DenseVegetation should be a shared event");
        host.EnterEventDebug(ev);

        EventView before = host.GetState().Event!;
        Assert.True(before.IsShared);
        Assert.Equal(2, before.Votes.Count);
        Assert.All(before.Votes, v => Assert.False(v.HasVoted)); // nobody has voted yet

        // Player 1 votes for the first option. The vote is recorded but the event does not resolve —
        // it waits for player 2 — and player 2 can see player 1's indicated choice.
        int chosenIndex = host.ListOptions(1uL)
            .First(o => o.Kind == OptionKind.ChooseEventOption).EventOptionIndex!.Value;
        host.Apply(host.ListOptions(1uL)
            .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == chosenIndex));

        GameState afterP1 = host.GetState();
        Assert.Equal(GamePhase.Event, afterP1.Phase); // still in the event, awaiting player 2
        EventVoteView p1Vote = afterP1.Event!.Votes.First(v => v.NetId == 1uL);
        EventVoteView p2Vote = afterP1.Event!.Votes.First(v => v.NetId == 2uL);
        Assert.True(p1Vote.HasVoted);
        Assert.Equal(chosenIndex, p1Vote.VotedOptionIndex);
        Assert.False(p2Vote.HasVoted); // player 2 still to vote
        _out.WriteLine($"after p1 vote: p1={p1Vote.VotedOptionIndex} p2 voted={p2Vote.HasVoted}");

        // Player 1 (already voted) is not offered another choice; player 2 still is.
        Assert.DoesNotContain(host.ListOptions(1uL), o => o.Kind == OptionKind.ChooseEventOption);
        Assert.Contains(host.ListOptions(2uL), o => o.Kind == OptionKind.ChooseEventOption);

        // Player 2 votes the same option, completing the tally — the shared option now resolves.
        host.Apply(host.ListOptions(2uL)
            .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == chosenIndex));

        // The event advanced (votes cleared on resolution; the event is no longer awaiting the initial
        // vote — DenseVegetation's "Rest" leads to a follow-up, others finish to the map).
        GameState resolved = host.GetState();
        _out.WriteLine($"resolved phase={resolved.Phase}");
        bool initialVotePending = resolved.Event is { } e
            && e.Votes.Any(v => v.HasVoted)
            && e.Options.Any(o => o.Index == chosenIndex);
        Assert.False(initialVotePending, "the shared option should have resolved after both votes");
    }

    /// <summary>Start a 2-player run and advance to the act map (resolving each player's own Neow).</summary>
    private static GameHost StartTwoPlayerOnMap(string seed)
    {
        GameHost host = GameHost.StartNewRun(seed, playerCount: 2);
        host.EnterFirstRoom();
        foreach (Player player in host.Run.Players)
        {
            int idx = BenignNeowOptionIndex(player);
            host.Apply(host.ListOptions(player.NetId)
                .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == idx));
        }
        return host;
    }

    [Fact]
    public async Task TwoPlayers_RestSite_EachRestsIndependently()
    {
        await Task.Run(RunRestSite).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunRestSite()
    {
        GameHost host = StartTwoPlayerOnMap("TESTSEED");
        // Damage both players so a rest (heal 30% max HP) is observable.
        foreach (Player p in host.Run.Players)
        {
            p.Creature.SetCurrentHpInternal(p.Creature.MaxHp / 2);
        }

        MapPointView rest = host.GetState().Map!.Points
            .First(p => p.PointType == MegaCrit.Sts2.Core.Map.MapPointType.RestSite);
        host.MoveTo(rest.Coord.ToMapCoord());
        Assert.Equal(GamePhase.RestSite, host.GetState().Phase);

        int p1Before = host.Run.Players[0].Creature.CurrentHp;
        int p2Before = host.Run.Players[1].Creature.CurrentHp;

        // Player 1 rests (heals). Player 2 still has their own rest options.
        host.Apply(host.ListOptions(1uL).First(o => o.Kind == OptionKind.ChooseRestOption && o.RestOptionId == "HEAL"));
        Assert.True(host.Run.Players[0].Creature.CurrentHp > p1Before, "player 1 should have healed");
        Assert.DoesNotContain(host.ListOptions(1uL), o => o.Kind == OptionKind.ChooseRestOption);
        Assert.Contains(host.ListOptions(2uL), o => o.Kind == OptionKind.ChooseRestOption);

        // Player 2 rests too. Now both have rested and the party can move on.
        host.Apply(host.ListOptions(2uL).First(o => o.Kind == OptionKind.ChooseRestOption && o.RestOptionId == "HEAL"));
        Assert.True(host.Run.Players[1].Creature.CurrentHp > p2Before, "player 2 should have healed");

        _out.WriteLine($"after rest: phase={host.GetState().Phase} p1={host.Run.Players[0].Creature.CurrentHp} p2={host.Run.Players[1].Creature.CurrentHp}");
        Assert.Equal(GamePhase.Map, host.GetState().Phase);
    }

    [Fact]
    public async Task TwoPlayers_Shop_EachBuysFromTheirOwnInventory()
    {
        await Task.Run(RunShop).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunShop()
    {
        GameHost host = StartTwoPlayerOnMap("TESTSEED");
        foreach (Player p in host.Run.Players)
        {
            p.Gold += 999; // afford anything
        }

        MapPointView shop = host.GetState().Map!.Points
            .First(p => p.PointType == MegaCrit.Sts2.Core.Map.MapPointType.Shop);
        host.MoveTo(shop.Coord.ToMapCoord());
        Assert.Equal(GamePhase.Shop, host.GetState().Phase);

        // Each player buys a card from *their own* inventory, spending their own gold.
        foreach (Player buyer in host.Run.Players)
        {
            int goldBefore = buyer.Gold;
            int deckBefore = buyer.Deck.Cards.Count;
            GameOption buy = host.ListOptions(buyer.NetId)
                .First(o => o.Kind == OptionKind.BuyShopItem && o.ShopItemType == "Card");
            _out.WriteLine($"p{buyer.NetId} buying {o_id(buy)} for {buy.ShopItemCost}");
            host.Apply(buy);

            Assert.True(buyer.Gold < goldBefore, $"player {buyer.NetId} should have spent gold");
            Assert.Equal(deckBefore + 1, buyer.Deck.Cards.Count); // the bought card went to *their* deck
        }
    }

    private static string o_id(GameOption o) => o.ShopItemId ?? "?";

    [Fact]
    public async Task TwoPlayers_TreasureChest_EachPicksTheirOwnRelic_WithVoteVisibility()
    {
        await Task.Run(RunTreasureVoting).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunTreasureVoting()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);
        host.EnterFirstRoom();
        foreach (Player player in host.Run.Players) // resolve each player's own Neow
        {
            int idx = BenignNeowOptionIndex(player);
            host.Apply(host.ListOptions(player.NetId)
                .First(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == idx));
        }

        // Jump the party straight to the act's treasure node (direct move; the vote path isn't needed
        // to set up the room). A multi-player chest offers one relic per player.
        MapPointView treasure = host.GetState().Map!.Points
            .First(p => p.PointType == MegaCrit.Sts2.Core.Map.MapPointType.Treasure);
        host.MoveTo(treasure.Coord.ToMapCoord());

        GameState atChest = host.GetState();
        Assert.Equal(GamePhase.Treasure, atChest.Phase);
        TreasureView view = atChest.Treasure!;
        _out.WriteLine($"chest relics=[{string.Join(",", view.Relics)}] votes={view.Votes.Count}");
        Assert.Equal(2, view.Relics.Count);          // one relic per player
        Assert.Equal(2, view.Votes.Count);
        Assert.All(view.Votes, v => Assert.False(v.HasVoted)); // auto-votes were reset; nobody picked

        string relic0 = view.Relics[0];
        string relic1 = view.Relics[1];

        // Player 1 picks relic 0. The pick is recorded but the chest does not resolve — it waits for
        // player 2 — and player 2 can see player 1's indicated pick.
        host.Apply(host.ListOptions(1uL)
            .First(o => o.Kind == OptionKind.TakeTreasureRelic && o.TreasureRelicIndex == 0));

        GameState afterP1 = host.GetState();
        Assert.Equal(GamePhase.Treasure, afterP1.Phase);
        TreasureVoteView p1 = afterP1.Treasure!.Votes.First(v => v.NetId == 1uL);
        TreasureVoteView p2 = afterP1.Treasure!.Votes.First(v => v.NetId == 2uL);
        Assert.True(p1.HasVoted);
        Assert.Equal(0, p1.VotedRelicIndex);
        Assert.False(p2.HasVoted);
        // Player 1 (picked) is no longer offered a choice; player 2 still is.
        Assert.DoesNotContain(host.ListOptions(1uL), o => o.Kind == OptionKind.TakeTreasureRelic);
        Assert.Contains(host.ListOptions(2uL), o => o.Kind == OptionKind.TakeTreasureRelic);

        // Player 2 picks relic 1 (a different relic — no conflict), completing the vote. The chest
        // resolves: each player gets the relic only they voted for, and the party returns to the map.
        host.Apply(host.ListOptions(2uL)
            .First(o => o.Kind == OptionKind.TakeTreasureRelic && o.TreasureRelicIndex == 1));

        GameState resolved = host.GetState();
        _out.WriteLine($"resolved phase={resolved.Phase} p1 relics=[{string.Join(",", resolved.Players[0].Relics)}] " +
                       $"p2 relics=[{string.Join(",", resolved.Players[1].Relics)}]");
        Assert.Null(resolved.Treasure);
        Assert.Equal(GamePhase.Map, resolved.Phase);
        Assert.Contains(relic0, resolved.Players[0].Relics); // player 1 got relic 0
        Assert.Contains(relic1, resolved.Players[1].Relics); // player 2 got relic 1
    }

    [Fact]
    public async Task TwoPlayers_PostCombatRewards_EachPlayerGetsTheirOwn()
    {
        await Task.Run(RunPostCombatRewards).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunPostCombatRewards()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);
        foreach (Player p in host.Run.Players)
        {
            p.Creature.SetMaxHpInternal(9999);
            p.Creature.SetCurrentHpInternal(9999);
        }

        host.EnterEncounterDebug(Act1FightsTests.ResolveEncounter("SlimesNormal"));
        DriveSharedCombatToEnd(host);

        // After the win, each alive player gets their own end-of-combat rewards, surfaced one player
        // at a time. Walk the reward screens, taking each owner's gold then proceeding.
        Assert.Equal(GamePhase.Reward, host.GetState().Phase);
        var goldTakenBy = new System.Collections.Generic.HashSet<ulong>();
        for (int guard = 0; guard < 10 && host.GetState().Phase == GamePhase.Reward; guard++)
        {
            GameOption? gold = host.ListOptions()
                .FirstOrDefault(o => o.Kind == OptionKind.TakeReward
                                     && o.Description.Contains("gold", StringComparison.Ordinal));
            if (gold is not null)
            {
                ulong owner = gold.PlayerId;
                int before = host.GetPlayerById(owner).Gold;
                host.Apply(gold);
                Assert.True(host.GetPlayerById(owner).Gold > before,
                    $"player {owner} should have gained reward gold");
                goldTakenBy.Add(owner);
            }
            host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.ProceedFromRewards));
        }

        _out.WriteLine($"gold taken by: [{string.Join(",", goldTakenBy)}] final phase={host.GetState().Phase}");
        // Both players received and took their own gold reward.
        Assert.Contains(1uL, goldTakenBy);
        Assert.Contains(2uL, goldTakenBy);
    }

    [Fact]
    public async Task TwoPlayers_BothActInOneCombat_AndTheFightResolves()
    {
        await Task.Run(RunTwoPlayerCombat).WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunTwoPlayerCombat()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", playerCount: 2);

        // Buff both players to a large HP pool so the fight is survivable, then drop straight into a
        // shared combat (bypassing the per-player Neow/map navigation, which is separate M6 work).
        foreach (Player p in host.Run.Players)
        {
            p.Creature.SetMaxHpInternal(9999);
            p.Creature.SetCurrentHpInternal(9999);
        }

        EncounterModel encounter = Act1FightsTests.ResolveEncounter("SlimesNormal");
        host.EnterEncounterDebug(encounter);
        Assert.True(host.InCombat, "expected both players to be in the shared combat");

        // Both players have their own combat state, hand and turn phase, both able to act.
        Assert.All(host.Run.Players, p => Assert.NotNull(p.PlayerCombatState));

        // Drive the shared fight. In the fake-multiplayer turn model both players act during the *same*
        // Play phase; ending any player ends the shared round (the enemy turn fires), so each step:
        //   1) resolve a pending mid-effect card choice (e.g. Silent's Survivor discard), else
        //   2) let any player still in Play play one of their cards, else
        //   3) no one has a card to play → end the turn once (local player) to trigger the enemy turn.
        // Stop when combat ends (won → rewards/map).
        bool bothPlayersGotToAct = false;
        for (int step = 0; step < 2000 && host.InCombat; step++)
        {
            if (host.GetState().Phase == GamePhase.Choice)
            {
                host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.SelectCards));
                continue;
            }

            GameOption? play = null;
            int playersWhoCanPlay = 0;
            foreach (Player player in host.Run.Players)
            {
                if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play)
                {
                    continue;
                }
                GameOption? candidate = host.ListOptions(player.NetId)
                    .FirstOrDefault(o => o.Kind == OptionKind.PlayCard);
                if (candidate is not null)
                {
                    playersWhoCanPlay++;
                    play ??= candidate;
                }
            }
            if (playersWhoCanPlay == 2)
            {
                bothPlayersGotToAct = true; // both players had a playable card in the same Play phase
            }

            if (play is not null)
            {
                host.Apply(play);
                continue;
            }

            // No player has a card to play this turn — end the shared round (local player), which
            // triggers the enemy turn (fake-multiplayer: any end ends the round).
            GameOption? end = host.ListOptions(host.Run.Players[0].NetId)
                .FirstOrDefault(o => o.Kind == OptionKind.EndTurn);
            if (end is null)
            {
                break; // nothing to do (transitioning) — avoid spinning
            }
            host.Apply(end);
        }

        Assert.False(host.InCombat, "the two-player fight should have resolved within the step budget");
        Assert.True(bothPlayersGotToAct, "both players should have been able to play cards in the shared combat");
        GameState end2 = host.GetState();
        _out.WriteLine($"ended phase={end2.Phase} p1={end2.Players[0].CurrentHp} p2={end2.Players[1].CurrentHp}");
        Assert.True(end2.Phase is GamePhase.Reward or GamePhase.Map or GamePhase.Choice,
            $"unexpected terminal phase {end2.Phase}");
    }

    /// <summary>
    /// Drive a shared two-player combat to its end. Each step: resolve a pending mid-effect card
    /// choice, else let any player still in Play play a card, else end the shared round (any end ends
    /// it in fake-multiplayer). Stops when combat ends.
    /// </summary>
    private static void DriveSharedCombatToEnd(GameHost host)
    {
        for (int step = 0; step < 2000 && host.InCombat; step++)
        {
            if (host.GetState().Phase == GamePhase.Choice)
            {
                host.Apply(host.ListOptions().First(o => o.Kind == OptionKind.SelectCards));
                continue;
            }

            GameOption? play = null;
            foreach (Player player in host.Run.Players)
            {
                if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play)
                {
                    continue;
                }
                play = host.ListOptions(player.NetId).FirstOrDefault(o => o.Kind == OptionKind.PlayCard);
                if (play is not null)
                {
                    break;
                }
            }
            if (play is not null)
            {
                host.Apply(play);
                continue;
            }

            GameOption? end = host.ListOptions(host.Run.Players[0].NetId)
                .FirstOrDefault(o => o.Kind == OptionKind.EndTurn);
            if (end is null)
            {
                break;
            }
            host.Apply(end);
        }
    }
}
