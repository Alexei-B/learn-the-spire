using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
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

    private static GameHost MoveIntoFirstCombat(string seed) => TestNav.MoveIntoFirstCombat(seed);
}
