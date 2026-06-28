using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

public sealed class CombatStateTests
{
    private readonly ITestOutputHelper _out;

    public CombatStateTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Combat_ExposesHandEnemiesAndIntents()
    {
        // Resolve the opening Neow ancient event, then move into the first combat.
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        CombatState combat = host.Combat!;

        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        _out.WriteLine($"Round={combat.RoundNumber} side={combat.CurrentSide} energy={pcs.Energy}/{pcs.MaxEnergy}");
        _out.WriteLine($"Hand ({pcs.Hand.Cards.Count}): {string.Join(", ", pcs.Hand.Cards.Select(c => c.Id))}");
        _out.WriteLine($"Draw={pcs.DrawPile.Cards.Count} Discard={pcs.DiscardPile.Cards.Count} Exhaust={pcs.ExhaustPile.Cards.Count}");

        foreach (Creature enemy in combat.Enemies)
        {
            string intents = enemy.Monster is null
                ? "?"
                : string.Join("+", enemy.Monster.NextMove.Intents.Select(i => i.GetType().Name));
            _out.WriteLine($"Enemy {enemy.Monster?.Id} hp={enemy.CurrentHp}/{enemy.MaxHp} block={enemy.Block} intent=[{intents}]");
        }

        // Playability of each hand card.
        foreach (CardModel card in pcs.Hand.Cards)
        {
            bool canPlay = card.CanPlay(out var reason, out _);
            _out.WriteLine($"  card {card.Id} canPlay={canPlay} reason={reason}");
        }

        Assert.NotEmpty(pcs.Hand.Cards);
        Assert.NotEmpty(combat.Enemies);
        Assert.True(pcs.MaxEnergy > 0);
    }
}
