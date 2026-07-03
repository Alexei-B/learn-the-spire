using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Character-specific combat-strategy behaviour of <see cref="RulesDecisionEngine"/>: the Necrobinder's
/// Osty summon counts as block, and the auto-play policy ends the turn when only unplayable junk is left
/// even with energy to spare.
/// </summary>
public sealed class NecrobinderStrategyTests
{
    private readonly ITestOutputHelper _out;
    private readonly RulesDecisionEngine _engine = new();
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
    public void OstyAttack_SurfacesCalculatedDamage_ScaledByOstyHp()
    {
        GameHost host = StartCombatAs("Necrobinder");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;

        // Unleash (CardTag.OstyAttack) deals 6 + Osty's current HP, carried in the CalculatedDamage var
        // rather than the plain Damage var. The projection must still surface it as the card's damage so
        // it shows after the card name and the auto-play strategy can rank it against plain strikes.
        pcs.Hand.AddInternal(combat.CreateCard<Unleash>(player));
        int ostyHp = player.Osty!.CurrentHp;
        Assert.True(ostyHp > 0, "test presumes a live Osty");

        CardView unleash = host.ListOptions()
            .Where(o => o.Kind == OptionKind.PlayCard && o.Card?.CardId == "UNLEASH")
            .Select(o => o.Card!)
            .First();

        _out.WriteLine($"Unleash damage={unleash.Damage}, base={unleash.BaseDamage}, ostyHp={ostyHp}");
        Assert.NotNull(unleash.Damage);
        // 6 printed base + Osty's HP; strictly greater than the printed base, so it clearly out-damages a
        // plain 6-damage strike whenever Osty has any HP.
        Assert.True(unleash.Damage >= 6 + ostyHp,
            $"Unleash should scale with Osty HP: expected >= {6 + ostyHp}, got {unleash.Damage}.");
    }

    [Fact]
    public void Defense_CountsLiveOstyHpAsBufferAgainstIncomingAttacks()
    {
        GameHost host = StartCombatAs("Necrobinder");
        CombatState combat = host.Combat!;
        Player player = combat.Players.Single();
        PlayerCombatState pcs = player.PlayerCombatState!;
        MegaCrit.Sts2.Core.Entities.Creatures.Creature osty = player.Osty!;

        // A live Osty soaks incoming (powered) attacks: they land on Osty and only damage beyond its HP
        // spills onto the player, so Osty's HP is a buffer on top of the player's block. The strategy must
        // account for it — otherwise it blocks against damage Osty already absorbs.
        GameState pre = host.GetState();
        int incoming = pre.Combat!.Enemies
            .SelectMany(e => e.Intents)
            .Where(i => i.Damage is not null)
            .Sum(i => i.Damage!.Value * (i.Hits ?? 1));
        Assert.True(incoming > 0, "test presumes the enemy telegraphs an attack this turn");

        // Leave a clean either/or in hand: one efficient block card and one attack.
        foreach (CardModel c in pcs.Hand.Cards.ToList())
        {
            pcs.Hand.RemoveInternal(c);
        }
        pcs.Hand.AddInternal(combat.CreateCard<DefendNecrobinder>(player)); // 5 block, Skill
        pcs.Hand.AddInternal(combat.CreateCard<StrikeNecrobinder>(player)); // 6 damage, Attack

        // Osty big enough to eat the whole telegraphed hit → blocking is wasted, so attack instead.
        osty.SetMaxHpInternal(incoming + 100);
        osty.SetCurrentHpInternal(incoming + 100);
        GameOption? withBigOsty = _engine.Recommend(host.GetState(), host.ListOptions());
        _out.WriteLine($"incoming={incoming}, bigOsty pick={withBigOsty?.Card?.Type.ToString() ?? "null"}");
        Assert.NotNull(withBigOsty);
        Assert.Equal(CardType.Attack, withBigOsty!.Card!.Type);

        // Shrink Osty to 1 HP so it no longer covers the hit. With the buffer gone the block card is chosen
        // whenever it's efficient (unblocked >= 4 for a 5-block card) — proving the buffer, not a constant,
        // drives the decision. (If this seed's hit is tiny, blocking stays inefficient and we still attack.)
        osty.SetMaxHpInternal(1);
        osty.SetCurrentHpInternal(1);
        bool blockIsEfficient = (incoming - 1) * 5 >= 5 * 4;
        GameOption? withTinyOsty = _engine.Recommend(host.GetState(), host.ListOptions());
        _out.WriteLine($"tinyOsty pick={withTinyOsty?.Card?.Type.ToString() ?? "null"}, blockEfficient={blockIsEfficient}");
        Assert.NotNull(withTinyOsty);
        Assert.Equal(blockIsEfficient ? CardType.Skill : CardType.Attack, withTinyOsty!.Card!.Type);
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

        GameOption? pick = _engine.Recommend(state, options);
        _out.WriteLine($"pick={pick?.Kind.ToString() ?? "null"}");
        Assert.NotNull(pick);
        Assert.Equal(OptionKind.EndTurn, pick!.Kind);
    }
}
