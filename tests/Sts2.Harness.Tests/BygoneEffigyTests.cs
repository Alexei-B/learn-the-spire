using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Encounters;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Runs;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Regression guard for the BygoneEffigy elite, whose enemy turn used to silently fault the
/// headless pump. The monster's wake move calls <c>TalkCmd.Play</c>, which reads
/// <c>SaveManager.Instance.PrefsSave.FastMode</c>; the harness boot never initialized the prefs
/// save, so that getter returned null and the move threw an (swallowed) NullReferenceException —
/// the enemy turn never completed and <see cref="GameHost.EndTurn"/> timed out. With prefs now
/// initialized at boot, the elite's sleep → wake (+10 Strength) → slash cycle resolves through the
/// public option API.
/// </summary>
public sealed class BygoneEffigyRepro
{
    private readonly ITestOutputHelper _out;

    public BygoneEffigyRepro(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task BygoneEffigy_EnemyTurn_ResolvesThroughSleepWakeSlash()
    {
        var t = Task.Run(Run);
        await t.WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        // Buff to a large HP pool so the elite's slash can't kill us before the cycle plays out.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        EncounterModel encounter = ModelDb.Encounter<BygoneEffigyElite>().ToMutable();
        Pump(RunManager.Instance.EnterRoomDebug(
            MegaCrit.Sts2.Core.Rooms.RoomType.Elite,
            model: encounter,
            showTransition: false));

        Assert.True(host.InCombat, "expected to be in combat with the elite");
        Creature effigy = host.Combat!.Enemies.Single();

        // Turn 0: SLEEP — the elite does nothing and gains no Strength.
        host.EndTurn(host.Run.Players[0]);
        Assert.True(host.InCombat);
        Assert.Equal(0, effigy.GetPowerAmount<StrengthPower>());

        // Turn 1: WAKE — the elite buffs itself by 10 Strength (the move that used to fault).
        host.EndTurn(host.Run.Players[0]);
        Assert.True(host.InCombat);
        int strength = effigy.GetPowerAmount<StrengthPower>();
        _out.WriteLine($"after wake: effigy strength={strength}");
        Assert.Equal(10, strength);

        // Turn 2: SLASH — the elite attacks; our big HP pool drops by the slash damage.
        int hpBefore = host.Run.Players[0].Creature.CurrentHp;
        host.EndTurn(host.Run.Players[0]);
        int hpAfter = host.Run.Players[0].Creature.CurrentHp;
        _out.WriteLine($"after slash: playerHp {hpBefore} -> {hpAfter}");
        Assert.True(hpAfter < hpBefore, "expected the elite's slash to deal damage");
    }

    private static T Pump<T>(Task<T> task) => task.GetAwaiter().GetResult();
}
