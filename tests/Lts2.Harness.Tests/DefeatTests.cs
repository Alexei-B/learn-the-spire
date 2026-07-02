using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.ValueProps;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Defeat and health-reducing events: dying ends the run and leaves no legal action (bar the menu),
/// exactly-lethal damage counts as death, and a self-inflicted event choice may never reduce you to
/// 0 HP or below (the game gates such options via <c>EventOption.WillKillPlayer</c>, which is only
/// enforced UI/AI-side, so the harness must enforce it too).
/// </summary>
public sealed class DefeatTests
{
    private readonly ITestOutputHelper _out;

    public DefeatTests(ITestOutputHelper output) => _out = output;

    private static EventModel ResolveAnyEvent(string typeName) =>
        Enumerable.Range(0, ModelDb.ActsByIndex.Count)
            .SelectMany(i => ModelDb.ActsByIndex[i])
            .SelectMany(a => a.AllEvents)
            .Concat(ModelDb.AllSharedEvents)
            .First(e => e.GetType().Name == typeName);

    private static void KillWithUnblockable(Creature target)
    {
        var ctx = new ThrowingPlayerChoiceContext();
        CreatureCmd.Damage(ctx, target, 99999m, ValueProp.Unblockable | ValueProp.Unpowered, dealer: null, cardSource: null)
            .GetAwaiter().GetResult();
        RunManager.Instance.ActionExecutor.FinishedExecutingActions().GetAwaiter().GetResult();
        // A player death marks the combat as a pending loss; the combat flow turns that into an actual
        // end at its next win-condition check (after a card/turn). The direct-damage seam above skips
        // that, so trigger it the way real play would.
        CombatManager.Instance.CheckWinCondition().GetAwaiter().GetResult();
        RunManager.Instance.ActionExecutor.FinishedExecutingActions().GetAwaiter().GetResult();
    }

    [Fact]
    public async Task Defeat_InCombat_EndsRun_AndLeavesNoLegalAction()
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
            KillWithUnblockable(host.Combat!.Players.Single().Creature);

            GameState s = host.GetState();
            _out.WriteLine($"phase={s.Phase} gameOver={s.IsGameOver} victory={s.IsVictory} hp={s.Players[0].CurrentHp} options={host.ListOptions().Count}");
            Assert.True(s.IsGameOver, "the run should be over once the player is dead");
            Assert.False(s.IsVictory);
            Assert.Equal(GamePhase.GameOver, s.Phase);
            Assert.False(host.InCombat);
            Assert.Empty(host.ListOptions()); // nothing to do after death but start a new run
        }).WaitAsync(TimeSpan.FromSeconds(60));
    }

    [Fact]
    public async Task ExactlyLethalDamage_Kills()
    {
        await Task.Run(() =>
        {
            GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
            Creature player = host.Combat!.Players.Single().Creature;
            TestNav.SetHp(host, maxHp: player.MaxHp, currentHp: 12);

            var ctx = new ThrowingPlayerChoiceContext();
            CreatureCmd.Damage(ctx, player, 12m, ValueProp.Unblockable | ValueProp.Unpowered, dealer: null, cardSource: null)
                .GetAwaiter().GetResult();
            RunManager.Instance.ActionExecutor.FinishedExecutingActions().GetAwaiter().GetResult();

            GameState s = host.GetState();
            _out.WriteLine($"exactly-lethal: hp={s.Players[0].CurrentHp} dead={player.IsDead} gameOver={s.IsGameOver}");
            Assert.Equal(0, s.Players[0].CurrentHp);
            Assert.True(player.IsDead, "0 HP means dead");
            Assert.True(s.IsGameOver);
        }).WaitAsync(TimeSpan.FromSeconds(60));
    }

    [Theory]
    [InlineData("DollRoom")]
    [InlineData("SlipperyBridge")]
    public async Task Event_NeverOffersAChoiceThatWouldKillYou(string eventName)
    {
        await Task.Run(() => RunEventInvariant(eventName)).WaitAsync(TimeSpan.FromSeconds(120));
    }

    private void RunEventInvariant(string eventName)
    {
        // At very low HP, apply each still-offered option on a fresh run; none may kill the player —
        // a self-damage option that would be lethal must have been filtered out.
        int[] indices = WithFreshEvent(eventName, hp: 3, host =>
            host.ListOptions().Where(o => o.Kind == OptionKind.ChooseEventOption)
                .Select(o => o.EventOptionIndex!.Value).ToArray());

        foreach (int idx in indices)
        {
            WithFreshEvent(eventName, hp: 3, host =>
            {
                GameOption? opt = host.ListOptions()
                    .FirstOrDefault(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == idx);
                if (opt is null)
                {
                    return 0;
                }
                host.Apply(opt);
                GameState s = host.GetState();
                _out.WriteLine($"{eventName} opt {idx}: hp={s.Players[0].CurrentHp} gameOver={s.IsGameOver}");
                Assert.False(s.IsGameOver, $"{eventName} option {idx} killed the player from 3 HP");
                Assert.True(s.Players[0].CurrentHp > 0, $"{eventName} option {idx} dropped HP to {s.Players[0].CurrentHp}");
                return 0;
            });
        }
    }

    [Fact]
    public async Task Event_OffersTheDamagingOption_WhenHpIsHighEnough()
    {
        // Sanity that the gate is dynamic, not a blanket hide: DollRoom offers more choices at full HP
        // (both HP-cost options) than at 3 HP (where the lethal ones are filtered).
        await Task.Run(() =>
        {
            int atFull = WithFreshEvent("DollRoom", hp: 80,
                host => host.ListOptions().Count(o => o.Kind == OptionKind.ChooseEventOption));
            int atLow = WithFreshEvent("DollRoom", hp: 3,
                host => host.ListOptions().Count(o => o.Kind == OptionKind.ChooseEventOption));
            _out.WriteLine($"DollRoom options: full={atFull} low={atLow}");
            Assert.True(atFull > atLow, $"expected more options at full HP ({atFull}) than at 3 HP ({atLow})");
        }).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private static T WithFreshEvent<T>(string eventName, int hp, Func<GameHost, T> body)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 80, currentHp: hp);
        host.EnterEventDebug(ResolveAnyEvent(eventName));
        return body(host);
    }
}
