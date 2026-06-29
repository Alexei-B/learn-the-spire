using System.Linq;
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// M5 — ascension. The ascension level is a <see cref="GameHost.StartNewRun(string, int)"/> param
/// plumbed into <c>RunState</c>; the game's <c>AscensionManager</c> applies the per-level modifiers.
/// These tests validate the observable ones end-to-end — the projected level, the tighter potion
/// belt, the Ascender's Bane curse, the double final boss, enemy HP scaling, and that the levels
/// are cumulative (a level-N run has every modifier up to N).
/// </summary>
public sealed class AscensionTests
{
    private readonly ITestOutputHelper _out;

    public AscensionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void AscensionLevel_IsProjectedIntoTheReadModel()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED", ascension: 5);
        Assert.Equal(5, host.GetState().AscensionLevel);

        host = GameHost.StartNewRun(seed: "TESTSEED", ascension: 0);
        Assert.Equal(0, host.GetState().AscensionLevel);
    }

    [Fact]
    public void HasAscension_IsCumulativeUpToTheRunsLevel()
    {
        // A level-5 run has every modifier with level <= 5 active and none above it.
        GameHost.StartNewRun(seed: "TESTSEED", ascension: 5);

        Assert.True(RunManager.Instance.HasAscension(AscensionLevel.SwarmingElites)); // 1
        Assert.True(RunManager.Instance.HasAscension(AscensionLevel.TightBelt));      // 4
        Assert.True(RunManager.Instance.HasAscension(AscensionLevel.AscendersBane));  // 5
        Assert.False(RunManager.Instance.HasAscension(AscensionLevel.Inflation));     // 6
        Assert.False(RunManager.Instance.HasAscension(AscensionLevel.DoubleBoss));    // 10
    }

    [Fact]
    public void TightBelt_ShrinksThePotionBeltByOne()
    {
        // TightBelt is level 4: the player carries one fewer potion slot.
        int slotsAtZero = GameHost.StartNewRun(seed: "TESTSEED", ascension: 0)
            .GetState().Players[0].Potions.Count;
        int slotsAtFour = GameHost.StartNewRun(seed: "TESTSEED", ascension: 4)
            .GetState().Players[0].Potions.Count;

        _out.WriteLine($"potion slots: A0={slotsAtZero} A4={slotsAtFour}");
        Assert.Equal(slotsAtZero - 1, slotsAtFour);
    }

    [Fact]
    public void AscendersBane_AddsACurseToTheStartingDeck()
    {
        // AscendersBane is level 5: a single eternal curse is added to the starting deck.
        var deckAtZero = GameHost.StartNewRun(seed: "TESTSEED", ascension: 0)
            .GetState().Players[0].Deck;
        var deckAtFive = GameHost.StartNewRun(seed: "TESTSEED", ascension: 5)
            .GetState().Players[0].Deck;

        Assert.DoesNotContain(deckAtZero, c => c.Type == CardType.Curse);
        Assert.Equal(deckAtZero.Count + 1, deckAtFive.Count);
        var curses = deckAtFive.Where(c => c.Type == CardType.Curse).ToList();
        _out.WriteLine($"Ascender's Bane card id: {curses.SingleOrDefault()?.CardId}");
        Assert.Single(curses);
    }

    [Fact]
    public void DoubleBoss_GivesTheFinalActASecondBossEncounter()
    {
        // DoubleBoss is level 10: only the final act gains a (distinct) second boss encounter.
        RunState atZero = GameHost.StartNewRun(seed: "TESTSEED", ascension: 0).Run;
        Assert.All(atZero.Acts, act => Assert.Null(act.SecondBossEncounter));

        RunState atTen = GameHost.StartNewRun(seed: "TESTSEED", ascension: 10).Run;
        for (int i = 0; i < atTen.Acts.Count - 1; i++)
        {
            Assert.Null(atTen.Acts[i].SecondBossEncounter);
        }
        ActModel finalAct = atTen.Acts[^1];
        Assert.NotNull(finalAct.SecondBossEncounter);
        Assert.NotEqual(finalAct.BossEncounter.Id, finalAct.SecondBossEncounter!.Id);
        _out.WriteLine($"final-act bosses: {finalAct.BossEncounter.Id.Entry} + {finalAct.SecondBossEncounter.Id.Entry}");
    }

    [Fact]
    public void ToughEnemies_RaisesMonsterInitialHp()
    {
        // ToughEnemies is level 8: a monster's initial-HP range shifts up. The model reads the
        // active run's ascension via AscensionHelper, so the same monster reports more HP at A8.
        int hpAtZero = WrigglerMinHpUnderAscension(0);
        int hpAtEight = WrigglerMinHpUnderAscension(8);

        _out.WriteLine($"Wriggler MinInitialHp: A0={hpAtZero} A8={hpAtEight}");
        Assert.True(hpAtEight > hpAtZero,
            $"expected ToughEnemies to raise initial HP (A0={hpAtZero}, A8={hpAtEight})");
    }

    private static int WrigglerMinHpUnderAscension(int ascension)
    {
        GameHost.StartNewRun(seed: "TESTSEED", ascension: ascension);
        MonsterModel wriggler = ModelDb.Monsters.First(m => m.GetType().Name == "Wriggler");
        return wriggler.MinInitialHp;
    }
}
