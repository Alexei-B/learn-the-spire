using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using Lts2.Harness;
using Lts2.Tui;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Character-specific combat-strategy behaviour: the Necrobinder's Osty summon counts as block, and
/// the auto-play policy ends the turn when only unplayable junk is left even with energy to spare.
/// </summary>
public sealed class NecrobinderStrategyTests
{
    private readonly ITestOutputHelper _out;
    public NecrobinderStrategyTests(ITestOutputHelper output) => _out = output;

    private static GameHost StartCombatAs(string characterTypeName)
    {
        CharacterModel ch = ModelDb.AllCharacters.First(c => c.GetType().Name == characterTypeName);
        GameHost host = GameHost.StartNewRun("TESTSEED", new[] { ch });
        host.EnterFirstRoom();
        TestNav.ResolveOpeningAncient(host);
        GameOption move = host.ListOptions().First(o => o.Kind == OptionKind.MoveTo);
        host.Apply(move);
        Assert.True(host.InCombat, $"expected {characterTypeName} to land in combat");
        return host;
    }

    [Fact]
    public void SummonCards_SurfaceSummonAsBlockEquivalent()
    {
        GameHost host = StartCombatAs("Necrobinder");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        pcs.Hand.AddInternal(combat.CreateCard<Bodyguard>(player));
        pcs.Hand.AddInternal(combat.CreateCard<Invoke>(player));

        var byId = host.ListOptions()
            .Where(o => o.Kind == OptionKind.PlayCard && o.Card is not null)
            .GroupBy(o => o.Card!.CardId)
            .ToDictionary(g => g.Key, g => g.First().Card!);

        _out.WriteLine($"Bodyguard summon={byId["BODYGUARD"].Summon}, Invoke summon={byId["INVOKE"].Summon}");
        Assert.True(byId["BODYGUARD"].Summon is > 0, "Bodyguard's summon should surface as block-equivalent.");
        Assert.True(byId["INVOKE"].Summon is > 0, "Invoke's summon should surface as block-equivalent.");
    }

    [Fact]
    public void AutoPlay_EndsTurn_WhenOnlyUnplayableJunkRemains_DespiteEnergy()
    {
        GameHost host = StartCombatAs("Necrobinder");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        // Empty the hand, then leave only a playable-but-worthless status card (Slimed). Energy is still
        // full at the start of the turn, so this is the "energy left but nothing worth playing" case.
        foreach (CardModel c in pcs.Hand.Cards.ToList())
        {
            pcs.Hand.RemoveInternal(c);
        }
        pcs.Hand.AddInternal(combat.CreateCard<Slimed>(player));

        GameState state = host.GetState();
        var options = host.ListOptions();
        Assert.True(state.Players[0].CombatState!.Energy > 0, "test presumes energy remains");
        Assert.Contains(options, o => o.Kind == OptionKind.EndTurn);

        GameOption? pick = CombatStrategy.ChooseDefaultMove(state, options);
        _out.WriteLine($"pick={pick?.Kind.ToString() ?? "null"}");
        Assert.NotNull(pick);
        Assert.Equal(OptionKind.EndTurn, pick!.Kind);
    }
}
