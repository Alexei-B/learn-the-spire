using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises choice-context injection: a mid-effect card selection (here a Discovery's
/// "discover a card" choice) surfaces through <see cref="GameHost.ListOptions(ulong)"/> as
/// <see cref="OptionKind.SelectCards"/> options and is resolved by <see cref="GameHost.Apply"/>,
/// instead of blocking on a UI screen.
/// </summary>
public sealed class ChoiceInjectionTests
{
    private readonly ITestOutputHelper _out;

    public ChoiceInjectionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Discovery_SurfacesChoice_AndApplyResolvesIt()
    {
        await Task.Run(RunDiscoveryChoice).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void RunDiscoveryChoice()
    {
        GameHost host = MoveIntoFirstCombat("TESTSEED");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        // Inject a Discovery into the hand: cost 1, Self target, "discover a card" on play.
        Discovery discovery = combat.CreateCard<Discovery>(player);
        pcs.Hand.AddInternal(discovery);
        Assert.True(discovery.CanPlay(), "the injected Discovery should be playable");
        int energyBefore = pcs.Energy;

        // Playing it triggers the discover choice; PlayCard returns once the effect blocks on it.
        Assert.True(host.PlayCard(discovery, target: null));

        GameState atChoice = host.GetState();
        _out.WriteLine($"phase={atChoice.Phase} pending={atChoice.PendingChoice?.Options.Count} min={atChoice.PendingChoice?.MinSelect} max={atChoice.PendingChoice?.MaxSelect}");
        Assert.Equal(GamePhase.Choice, atChoice.Phase);
        Assert.NotNull(atChoice.PendingChoice);
        Assert.Equal(3, atChoice.PendingChoice!.Options.Count);
        Assert.Equal(0, atChoice.PendingChoice.MinSelect);
        Assert.Equal(1, atChoice.PendingChoice.MaxSelect);

        // Options: one per discoverable card, plus a skip (the choice allows selecting none).
        var options = host.ListOptions();
        foreach (GameOption o in options)
        {
            _out.WriteLine($"{o.Kind}: {o.Description}");
        }
        Assert.Equal(4, options.Count);
        Assert.All(options, o => Assert.Equal(OptionKind.SelectCards, o.Kind));
        Assert.Single(options, o => o.SelectedCards!.Count == 0); // the skip option

        // Each single-card option carries its card view so the UI can render the card's name and
        // description (the skip option carries none).
        foreach (GameOption o in options.Where(o => o.SelectedCards!.Count == 1))
        {
            Assert.NotNull(o.Card);
            Assert.Equal(o.SelectedCards![0].CardId, o.Card!.CardId);
        }
        Assert.Null(options.Single(o => o.SelectedCards!.Count == 0).Card);

        // Choose the first offered card and resolve.
        GameOption pick = options.First(o => o.SelectedCards!.Count == 1);
        string chosenId = pick.SelectedCards![0].CardId;
        host.Apply(pick);

        GameState after = host.GetState();
        Assert.Null(after.PendingChoice);
        Assert.NotEqual(GamePhase.Choice, after.Phase);

        PlayerCombatView combatView = after.Players[0].CombatState!;
        _out.WriteLine($"chose {chosenId}; hand now: {string.Join(", ", combatView.Hand.Select(c => c.CardId))}");
        Assert.Contains(combatView.Hand, c => c.CardId == chosenId);
        Assert.Equal(energyBefore - 1, combatView.Energy); // Discovery cost 1 energy
    }

    [Fact]
    public async Task Skipping_AChoice_AddsNothing()
    {
        await Task.Run(RunDiscoverySkip).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void RunDiscoverySkip()
    {
        GameHost host = MoveIntoFirstCombat("TESTSEED");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        Discovery discovery = combat.CreateCard<Discovery>(player);
        pcs.Hand.AddInternal(discovery);
        int handCountBeforePlay = pcs.Hand.Cards.Count;

        Assert.True(host.PlayCard(discovery, target: null));
        Assert.Equal(GamePhase.Choice, host.GetState().Phase);

        // Resolve with the skip option (selects nothing).
        GameOption skip = host.ListOptions().Single(o => o.SelectedCards!.Count == 0);
        host.Apply(skip);

        GameState after = host.GetState();
        Assert.Null(after.PendingChoice);
        // No discovered card added; Discovery itself left the hand (exhausted), so the hand is
        // smaller than it was with the unplayed Discovery in it.
        Assert.True(after.Players[0].CombatState!.Hand.Count < handCountBeforePlay);
    }

    [Fact]
    public async Task Charge_MultiSelectChoice_ResolvesAnySubsetViaApplyCardChoice()
    {
        await Task.Run(RunChargeMultiSelect).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void RunChargeMultiSelect()
    {
        GameHost host = MoveIntoFirstCombat("TESTSEED");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        // CHARGE!! (the Regent's card) transforms 2 cards *chosen from the draw pile* — a genuine
        // multi-select (min=max=2). Need >2 draw-pile cards or the game auto-resolves without a choice.
        Charge charge = combat.CreateCard<Charge>(player);
        pcs.Hand.AddInternal(charge);
        var drawBefore = pcs.DrawPile.Cards.ToList();
        _out.WriteLine($"draw pile has {drawBefore.Count}: {string.Join(", ", drawBefore.Select(c => c.Id.Entry))}");
        Assert.True(drawBefore.Count > 2, "need >2 draw-pile cards for a real 2-of-N choice");
        Assert.True(charge.CanPlay(), "the injected Charge should be playable");

        Assert.True(host.PlayCard(charge, target: null));

        GameState atChoice = host.GetState();
        Assert.Equal(GamePhase.Choice, atChoice.Phase);
        Assert.NotNull(atChoice.PendingChoice);
        Assert.Equal(2, atChoice.PendingChoice!.MinSelect);
        Assert.Equal(2, atChoice.PendingChoice.MaxSelect);
        int n = atChoice.PendingChoice.Options.Count;
        Assert.Equal(drawBefore.Count, n); // one option per draw-pile card, in pile order

        // Validation: the count must be exactly 2, indices distinct and in range. None of these
        // resolve the choice — it stays pending until a valid selection is applied.
        Assert.Throws<ArgumentException>(() => host.ApplyCardChoice(new[] { 0 }));            // too few
        Assert.Throws<ArgumentException>(() => host.ApplyCardChoice(new[] { 0, 1, 2 }));      // too many
        Assert.Throws<ArgumentException>(() => host.ApplyCardChoice(new[] { 1, 1 }));         // duplicate
        Assert.Throws<ArgumentOutOfRangeException>(() => host.ApplyCardChoice(new[] { 0, n })); // out of range
        Assert.Equal(GamePhase.Choice, host.GetState().Phase);

        // Pick the *last two* cards — not the first two the fixed option path would have taken — to
        // prove any valid subset (not just the leading min) resolves.
        string bombId = combat.CreateCard<MinionDiveBomb>(player).Id.Entry;
        int bombsBefore = CountAcrossPiles(host.GetState().Players[0].CombatState!, bombId);
        host.ApplyCardChoice(new[] { n - 2, n - 1 });

        GameState after = host.GetState();
        Assert.Null(after.PendingChoice);
        Assert.NotEqual(GamePhase.Choice, after.Phase);

        // Charge transforms each chosen card into a MinionDiveBomb; picking two produced two of them,
        // confirming a two-card multi-select was resolved end-to-end.
        int bombsAfter = CountAcrossPiles(after.Players[0].CombatState!, bombId);
        _out.WriteLine($"MinionDiveBomb across piles: {bombsBefore} -> {bombsAfter}");
        Assert.Equal(bombsBefore + 2, bombsAfter);
    }

    private static int CountAcrossPiles(PlayerCombatView cs, string cardId) =>
        cs.Hand.Concat(cs.DrawPile).Concat(cs.DiscardPile).Concat(cs.ExhaustPile)
            .Count(c => c.CardId == cardId);

    private static GameHost MoveIntoFirstCombat(string seed) => TestNav.MoveIntoFirstCombat(seed);
}
