using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models.Cards;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Regression: an unplayable status/curse card in hand (Burn/Injury/Slimed) must not knock the whole
/// options list out — End Turn (and the other legal plays) must still be listed.
/// </summary>
public sealed class StatusCardOptionsTests
{
    private readonly ITestOutputHelper _out;

    public StatusCardOptionsTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void StatusCardInHand_StillListsEndTurn()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        Burn burn = combat.CreateCard<Burn>(player);
        pcs.Hand.AddInternal(burn);
        _out.WriteLine($"Hand: {string.Join(", ", pcs.Hand.Cards.Select(c => c.Id.Entry))}");

        var options = host.ListOptions().ToList();
        _out.WriteLine($"Options: {string.Join(" | ", options.Select(o => o.Kind + ":" + o.Description))}");

        Assert.Contains(options, o => o.Kind == OptionKind.EndTurn);
    }
}
