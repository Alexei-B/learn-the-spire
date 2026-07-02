using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// The "choose a bundle" selection (ScrollBoxes, a Neow relic that offers two 3-card bundles for the
/// deck). The game's <c>FromChooseABundleScreen</c> silently auto-takes bundles[0] under TestMode; a
/// Harmony patch routes it to the harness so the bundles surface as <see cref="GamePhase.BundleChoice"/>
/// options the agent actually picks between.
/// </summary>
public sealed class BundleChoiceTests
{
    private readonly ITestOutputHelper _out;

    public BundleChoiceTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task ScrollBoxes_SurfacesBundleChoice_AndApplyAddsChosenBundle()
    {
        await Task.Run(RunScrollBoxes).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void RunScrollBoxes()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        int deckBefore = host.GetState().Players[0].Deck.Count;

        // Granting ScrollBoxes runs its AfterObtained, which offers two card bundles. The obtain is
        // pumped until it suspends on that choice (rather than auto-taking bundle[0] as TestMode would).
        host.ObtainRelicDebug(ModelDb.Relic<ScrollBoxes>());

        GameState atChoice = host.GetState();
        _out.WriteLine($"phase={atChoice.Phase} bundles={atChoice.BundleChoice?.Bundles.Count}");
        Assert.Equal(GamePhase.BundleChoice, atChoice.Phase);
        Assert.NotNull(atChoice.BundleChoice);
        Assert.Equal(2, atChoice.BundleChoice!.Bundles.Count);
        Assert.All(atChoice.BundleChoice.Bundles, b => Assert.Equal(3, b.Count)); // 2 commons + 1 uncommon

        var options = host.ListOptions();
        Assert.Equal(2, options.Count);
        Assert.All(options, o => Assert.Equal(OptionKind.ChooseBundle, o.Kind));
        foreach (GameOption o in options)
        {
            Assert.NotNull(o.BundleCards);
            _out.WriteLine($"{o.Description}");
        }

        // Take the second bundle (not the first the TestMode shortcut would have auto-selected); its
        // exact cards should be added to the deck.
        GameOption pick = options[1];
        var chosenIds = pick.BundleCards!.Select(c => c.CardId).ToList();
        host.Apply(pick);

        GameState after = host.GetState();
        Assert.Null(after.BundleChoice);
        Assert.NotEqual(GamePhase.BundleChoice, after.Phase);

        // The chosen bundle's three cards are now in the deck.
        var deck = after.Players[0].Deck.Select(c => c.CardId).ToList();
        Assert.Equal(deckBefore + 3, deck.Count);
        foreach (string id in chosenIds)
        {
            Assert.Contains(id, deck);
        }
    }
}
