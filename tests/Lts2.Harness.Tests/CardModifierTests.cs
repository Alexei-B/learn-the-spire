using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Afflictions;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Enchantments;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Card modifiers applied by effects (an affliction like Bound, an enchantment, a granted Replay
/// count, a keyword such as Retain from Transfigure) surface on the card's read-model so the TUI can
/// show them in the hand, the piles, and the deck.
/// </summary>
public sealed class CardModifierTests
{
    private readonly ITestOutputHelper _out;

    public CardModifierTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void AppliedModifiers_SurfaceOnTheCardView()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        // Inject a unique card so we can find its view unambiguously, then pile modifiers onto it.
        Discovery card = combat.CreateCard<Discovery>(player);
        pcs.Hand.AddInternal(card);
        card.AfflictInternal(ModelDb.Affliction<Bound>().ToMutable(), 1);
        card.EnchantInternal(ModelDb.Enchantment<Imbued>().ToMutable(), 1);
        card.BaseReplayCount = 2;
        card.AddKeyword(CardKeyword.Retain);

        CardView view = host.GetState().Players[0].CombatState!.Hand.First(c => c.CardId == card.Id.Entry);
        _out.WriteLine($"affliction={view.AfflictionId} enchant={view.EnchantmentId} replay={view.ReplayCount} added=[{string.Join(",", view.AddedKeywords)}]");

        Assert.Equal("BOUND", view.AfflictionId);
        Assert.Equal("IMBUED", view.EnchantmentId);
        Assert.Equal(2, view.ReplayCount);
        Assert.Contains("Retain", view.AddedKeywords);
    }

    [Fact]
    public void APlainCard_HasNoModifiers()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        CardView strike = host.GetState().Players[0].CombatState!.Hand
            .First(c => c.Type == CardType.Attack);
        Assert.Null(strike.AfflictionId);
        Assert.Null(strike.EnchantmentId);
        Assert.Equal(0, strike.ReplayCount);
        Assert.Empty(strike.AddedKeywords);
    }
}
