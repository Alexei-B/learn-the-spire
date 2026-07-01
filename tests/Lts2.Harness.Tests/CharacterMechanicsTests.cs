using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Powers;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Combat read-model previews that the TUI colours: a card's actual attack damage / block after all
/// powers (self and target), the enemy intent's actual damage, the Regent's star power + per-card
/// star costs, and the Defect's orb slots and channeled orb values.
/// </summary>
public sealed class CharacterMechanicsTests
{
    private readonly ITestOutputHelper _out;

    public CharacterMechanicsTests(ITestOutputHelper output) => _out = output;

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

    private static void Apply<T>(Creature target, int amount) where T : PowerModel =>
        ModelDb.Power<T>().ToMutable().ApplyInternal(target, amount);

    [Fact]
    public void AttackDamage_ReflectsStrengthAndTargetVulnerable()
    {
        GameHost host = StartCombatAs("Ironclad");
        CombatState combat = host.Combat!;
        Creature player = combat.Players.Single().Creature;

        CardView strike = FirstStrikeOption(host).Card!;
        Assert.NotNull(strike.Damage);
        Assert.NotNull(strike.BaseDamage);
        // Baseline: no modifiers, so actual == printed.
        Assert.Equal(strike.BaseDamage, strike.Damage);
        int baseDmg = strike.BaseDamage!.Value;
        _out.WriteLine($"strike base damage {baseDmg}");

        // Strength buffs the attacker: actual damage climbs above the printed value.
        Apply<StrengthPower>(player, 3);
        CardView buffed = FirstStrikeOption(host).Card!;
        Assert.Equal(baseDmg + 3, buffed.Damage);
        Assert.True(buffed.Damage > buffed.BaseDamage);

        // Vulnerable on the specific target amplifies it further (the option is projected per target).
        GameOption strikeOpt = FirstStrikeOption(host);
        Creature enemy = combat.HittableEnemies.First(e => e.CombatId == strikeOpt.TargetCombatId);
        Apply<VulnerablePower>(enemy, 2);
        int vulnerable = FirstStrikeOption(host).Card!.Damage!.Value;
        _out.WriteLine($"strength {baseDmg + 3}, then vulnerable -> {vulnerable}");
        Assert.True(vulnerable > baseDmg + 3, $"vulnerable should raise damage above {baseDmg + 3}, got {vulnerable}");
    }

    [Fact]
    public void Block_ReflectsDexterityAndFrail()
    {
        GameHost host = StartCombatAs("Ironclad");
        Creature player = host.Combat!.Players.Single().Creature;

        CardView defend = FirstDefendOption(host).Card!;
        Assert.NotNull(defend.Block);
        Assert.Equal(defend.BaseBlock, defend.Block);
        int baseBlock = defend.BaseBlock!.Value;

        Apply<DexterityPower>(player, 2);
        Assert.Equal(baseBlock + 2, FirstDefendOption(host).Card!.Block);

        // Frail cuts block; net of +2 Dexterity it should drop below the buffed value.
        Apply<FrailPower>(player, 1);
        int frailed = FirstDefendOption(host).Card!.Block!.Value;
        _out.WriteLine($"base {baseBlock}, +dex {baseBlock + 2}, +frail -> {frailed}");
        Assert.True(frailed < baseBlock + 2, $"frail should reduce block below {baseBlock + 2}, got {frailed}");
    }

    [Fact]
    public void EnemyIntent_ExposesActualAndBaseDamage()
    {
        GameHost host = StartCombatAs("Ironclad");
        CombatView combat = host.GetState().Combat!;
        EnemyView enemy = combat.Enemies.First(e => e.Intents.Any(i => i.Damage is not null));
        IntentView attack = enemy.Intents.First(i => i.Damage is not null);
        Assert.NotNull(attack.BaseDamage);
        Assert.Equal(attack.BaseDamage, attack.Damage); // no modifiers yet

        // Vulnerable on the player raises the incoming damage above the base.
        Apply<VulnerablePower>(host.Combat!.Players.Single().Creature, 2);
        IntentView after = host.GetState().Combat!.Enemies
            .First(e => e.CombatId == enemy.CombatId).Intents.First(i => i.Damage is not null);
        _out.WriteLine($"intent base {after.BaseDamage} -> actual {after.Damage}");
        Assert.True(after.Damage > after.BaseDamage);
    }

    [Fact]
    public void Regent_HasStarPower_AndCardsReportStarCost()
    {
        GameHost host = StartCombatAs("Regent");
        PlayerCombatView cs = host.GetState().Players[0].CombatState!;
        _out.WriteLine($"regent stars={cs.Stars}");
        Assert.True(cs.Stars > 0, "the Regent starts combat with star power");

        // A star-cost card reports its cost through the read model (Comet costs stars).
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        Comet comet = combat.CreateCard<Comet>(player);
        player.PlayerCombatState!.Hand.AddInternal(comet);

        CardView view = host.GetState().Players[0].CombatState!.Hand.First(c => c.CardId == comet.Id.Entry);
        _out.WriteLine($"comet star cost {view.StarCost}");
        Assert.True(view.StarCost > 0);
    }

    [Fact]
    public void Defect_HasOrbSlots_AndChanneledOrbsExposeTheirValues()
    {
        GameHost host = StartCombatAs("Defect");
        PlayerCombatView cs = host.GetState().Players[0].CombatState!;
        _out.WriteLine($"defect orb slots={cs.OrbSlots} orbs=[{string.Join(",", cs.Orbs.Select(o => $"{o.OrbId} {o.PassiveValue}/{o.EvokeValue}"))}]");
        Assert.Equal(3, cs.OrbSlots);

        // Cracked Core channels a Lightning orb at combat start; its passive value is exposed.
        OrbView lightning = Assert.Single(cs.Orbs, o => o.OrbId == "LIGHTNING_ORB");
        Assert.Equal(3, lightning.PassiveValue);
        Assert.Equal(8, lightning.EvokeValue);
    }

    private static GameOption FirstStrikeOption(GameHost host) =>
        host.ListOptions().First(o => o.Kind == OptionKind.PlayCard && o.Card!.Type == CardType.Attack && o.Card.Damage is not null);

    private static GameOption FirstDefendOption(GameHost host) =>
        host.ListOptions().First(o => o.Kind == OptionKind.PlayCard && o.Card!.Block is not null);
}
