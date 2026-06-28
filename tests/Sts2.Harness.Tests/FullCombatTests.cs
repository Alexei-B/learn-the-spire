using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

public sealed class FullCombatTests
{
    private readonly ITestOutputHelper _out;

    public FullCombatTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task GreedyAutoCombat_FinishesTheFirstFight()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();
        var rs = host.Run;
        var firstMonster = (rs.CurrentMapPoint?.Children ?? rs.Map.StartingMapPoint.Children)
            .OrderBy(p => p.coord.col).First();
        host.MoveTo(firstMonster.coord);
        Assert.True(host.InCombat);

        var t = Task.Run(() => PlayUntilCombatEnds(host, maxTurns: 50));
        int turns = await t.WaitAsync(TimeSpan.FromSeconds(60));

        _out.WriteLine($"Combat ended after {turns} turns. inCombat={host.InCombat} room={rs.CurrentRoom?.GetType().Name} playerHp={rs.Players[0].Creature.CurrentHp}/{rs.Players[0].Creature.MaxHp}");

        Assert.False(host.InCombat, "combat should have ended");
        // Player should have survived this easy opening fight.
        Assert.True(rs.Players[0].Creature.IsAlive, "player should have survived the first fight");
    }

    private int PlayUntilCombatEnds(GameHost host, int maxTurns)
    {
        int turns = 0;
        while (host.InCombat && turns < maxTurns)
        {
            CombatState combat = host.Combat!;
            Player player = combat.Players.Single();
            PlayerCombatState pcs = player.PlayerCombatState!;

            // Greedily play every playable card this turn.
            bool playedSomething = true;
            int guard = 0;
            while (playedSomething && host.InCombat && guard++ < 100)
            {
                playedSomething = false;
                foreach (CardModel card in pcs.Hand.Cards.ToList())
                {
                    if (!card.CanPlay(out _, out _))
                    {
                        continue;
                    }
                    Creature? target = null;
                    if (card.TargetType == TargetType.AnyEnemy)
                    {
                        target = combat.HittableEnemies.FirstOrDefault();
                        if (target == null)
                        {
                            continue;
                        }
                    }
                    if (host.PlayCard(card, target))
                    {
                        playedSomething = true;
                        break;
                    }
                }
            }

            if (host.InCombat)
            {
                host.EndTurn(player);
            }
            turns++;
        }
        return turns;
    }
}
