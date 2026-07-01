using System.Linq;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Exercises potion actions (M2): using a potion (targeted/untargeted, in and out of combat) and
/// discarding one. Potions are given to the player directly; actions resolve through the faithful
/// manual-use path (<c>PotionModel.EnqueueManualUse</c>) / <c>DiscardPotionGameAction</c>, the same
/// the UI's potion popup drives. Usage gating mirrors the game: AnyTime potions work anywhere,
/// CombatOnly only in combat; discard is always available.
/// </summary>
public sealed class PotionTests
{
    private readonly ITestOutputHelper _out;

    public PotionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Potion_AnyTime_UsedOutOfCombat_HealsAndEmptiesSlot()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 80, currentHp: 40);
        PotionModel potion = TestNav.GivePotion(host, "BloodPotion"); // AnyTime, heals 20% max HP

        GameState before = host.GetState();
        Assert.Contains(before.Players[0].Potions, p => p == potion.Id.Entry);

        // The potion surfaces as a usable option on the map (AnyTime).
        GameOption use = host.ListOptions().First(o =>
            o.Kind == OptionKind.UsePotion && o.PotionId == potion.Id.Entry);
        _out.WriteLine($"using {use.PotionId} at hp={before.Players[0].CurrentHp}");
        host.Apply(use);

        GameState after = host.GetState();
        _out.WriteLine($"hp {before.Players[0].CurrentHp} -> {after.Players[0].CurrentHp}");
        // Heal is 20% of 80 = 16; the slot is now empty.
        Assert.Equal(56, after.Players[0].CurrentHp);
        Assert.DoesNotContain(after.Players[0].Potions, p => p == potion.Id.Entry);
        Assert.DoesNotContain(host.ListOptions(), o => o.Kind == OptionKind.UsePotion);
    }

    [Fact]
    public void Potion_CombatOnly_NotUsableOutOfCombat_ButDiscardable()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        PotionModel potion = TestNav.GivePotion(host, "BlockPotion"); // CombatOnly

        var options = host.ListOptions();
        // Out of combat a CombatOnly potion offers no use option, but can still be discarded.
        Assert.DoesNotContain(options, o => o.Kind == OptionKind.UsePotion && o.PotionId == potion.Id.Entry);
        Assert.Contains(options, o => o.Kind == OptionKind.DiscardPotion && o.PotionId == potion.Id.Entry);
    }

    [Fact]
    public void Potion_Discard_EmptiesSlot()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        PotionModel potion = TestNav.GivePotion(host, "FirePotion");
        Assert.Contains(host.GetState().Players[0].Potions, p => p == potion.Id.Entry);

        GameOption discard = host.ListOptions().First(o =>
            o.Kind == OptionKind.DiscardPotion && o.PotionId == potion.Id.Entry);
        host.Apply(discard);

        GameState after = host.GetState();
        _out.WriteLine($"after discard potions=[{string.Join(",", after.Players[0].Potions.Where(p => p is not null))}]");
        Assert.DoesNotContain(after.Players[0].Potions, p => p == potion.Id.Entry);
    }

    [Fact]
    public void Potion_Targeted_UsedInCombat_DamagesChosenEnemy()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        TestNav.SetHp(host, maxHp: 200, currentHp: 200); // survive enemy turn comfortably
        PotionModel potion = TestNav.GivePotion(host, "FirePotion"); // CombatOnly, AnyEnemy, 20 dmg

        // A targeted potion expands to one use option per hittable enemy.
        var useOptions = host.ListOptions()
            .Where(o => o.Kind == OptionKind.UsePotion && o.PotionId == potion.Id.Entry)
            .ToList();
        Assert.NotEmpty(useOptions);
        Assert.All(useOptions, o => Assert.NotNull(o.TargetCombatId));

        GameOption use = useOptions.First();
        uint targetId = use.TargetCombatId!.Value;
        int hpBefore = host.GetState().Combat!.Enemies.First(e => e.CombatId == targetId).CurrentHp;
        _out.WriteLine($"throwing {potion.Id.Entry} at enemy {targetId} (hp {hpBefore})");
        host.Apply(use);

        GameState after = host.GetState();
        // Combat may have ended if the potion killed the only enemy; otherwise the target lost HP.
        EnemyView? target = after.Combat?.Enemies.FirstOrDefault(e => e.CombatId == targetId);
        if (target is not null)
        {
            _out.WriteLine($"enemy hp {hpBefore} -> {target.CurrentHp}");
            Assert.True(target.CurrentHp < hpBefore || !target.IsHittable,
                $"expected the thrown potion to damage the enemy (was {hpBefore}, now {target.CurrentHp})");
        }
        // The potion was consumed regardless.
        Assert.DoesNotContain(after.Players[0].Potions, p => p == potion.Id.Entry);
    }
}
