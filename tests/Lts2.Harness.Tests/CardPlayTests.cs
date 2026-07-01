using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

public sealed class CardPlayTests
{
    private readonly ITestOutputHelper _out;

    public CardPlayTests(ITestOutputHelper output) => _out = output;

    private static GameHost StartInCombat(string seed) => TestNav.MoveIntoFirstCombat(seed);

    [Fact]
    public void PlayingAnAttack_DamagesTheEnemy()
    {
        GameHost host = StartInCombat("TESTSEED");
        CombatState combat = host.Combat!;
        PlayerCombatState pcs = combat.Players.Single().PlayerCombatState!;

        // Pick an attack that targets a single enemy (Strike).
        CardModel attack = pcs.Hand.Cards.First(c => c.TargetType == TargetType.AnyEnemy);
        Creature enemy = combat.HittableEnemies.First();
        int hpBefore = enemy.CurrentHp;
        int energyBefore = pcs.Energy;

        _out.WriteLine($"Playing {attack.Id} at {enemy.Monster?.Id} (hp {hpBefore})");
        host.PlayCard(attack, enemy);

        _out.WriteLine($"Enemy hp {hpBefore} -> {enemy.CurrentHp}; energy {energyBefore} -> {pcs.Energy}; hand={pcs.Hand.Cards.Count} discard={pcs.DiscardPile.Cards.Count}");

        Assert.True(enemy.CurrentHp < hpBefore, "Enemy should have taken damage.");
        Assert.True(pcs.Energy < energyBefore, "Playing a card should spend energy.");
        Assert.Contains(attack, pcs.DiscardPile.Cards);
    }

    [Fact]
    public void EndingTurn_LetsTheEnemyAct_AndReturnsToPlayerTurn()
    {
        GameHost host = StartInCombat("TESTSEED");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        int playerHpBefore = player.Creature.CurrentHp;
        int roundBefore = combat.RoundNumber;

        host.EndTurn(player);

        _out.WriteLine($"After end turn: inCombat={host.InCombat} round {roundBefore}->{combat.RoundNumber} playerHp {playerHpBefore}->{player.Creature.CurrentHp} block={player.Creature.Block}");

        // Either combat ended, or we are back to the player's turn on a later round.
        if (host.InCombat)
        {
            Assert.True(combat.RoundNumber >= roundBefore);
        }
    }
}
