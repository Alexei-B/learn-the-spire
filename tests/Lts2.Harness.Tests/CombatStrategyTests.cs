using System.Collections.Generic;
using System.Linq;
using Lts2.Harness;
using Lts2.Tui;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Drives a real first combat entirely by <see cref="CombatStrategy.ChooseDefaultMove"/> (the TUI's
/// Tab auto-play). Verifies the policy only ever returns a legal, listed move, that its choices honour
/// the documented rules (block only when it would take damage and the block is ≥80% used; attacks aim
/// at the lowest-health enemy; it only ends the turn when out of energy), and that it plays the fight
/// to a finish.
/// </summary>
public sealed class CombatStrategyTests
{
    private readonly ITestOutputHelper _out;

    public CombatStrategyTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void DefaultStrategy_PlaysOutCombat_LegallyAndByTheRules()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");

        bool sawPlay = false;
        for (int step = 0; step < 80; step++)
        {
            GameState state = host.GetState();
            if (state.IsGameOver || state.Phase is not (GamePhase.Combat or GamePhase.Choice))
            {
                break;
            }

            IReadOnlyList<GameOption> options = host.ListOptions();

            // A mid-effect card choice can surface (e.g. a discover): the policy doesn't model choices,
            // so resolve it minimally to keep the fight moving.
            if (state.Phase == GamePhase.Choice)
            {
                GameOption resolve = options.First(o => o.Kind == OptionKind.SelectCards);
                host.Apply(resolve);
                continue;
            }

            GameOption? pick = CombatStrategy.ChooseDefaultMove(state, options);
            if (pick is null)
            {
                // The policy declined (it has energy but only cards it doesn't act on). That must not
                // happen when out of energy — then it is required to end the turn.
                int idle = state.Players[0].CombatState?.Energy ?? 0;
                Assert.True(idle > 0, "Policy returned no move while out of energy — it should end the turn.");
                host.EndTurn(host.Combat!.Players.Single());
                continue;
            }

            Assert.Contains(pick, options); // never invents an illegal move
            AssertObeysRules(state, options, pick);
            if (pick.Kind == OptionKind.PlayCard)
            {
                sawPlay = true;
            }

            _out.WriteLine($"[{step}] {pick.Kind} {pick.Description}");
            host.Apply(pick);
        }

        GameState end = host.GetState();
        _out.WriteLine($"Ended: phase={end.Phase} gameOver={end.IsGameOver} hp={end.Players[0].CurrentHp}");
        Assert.False(end.Phase is GamePhase.Combat or GamePhase.Choice, "The strategy should finish the combat.");
        Assert.True(sawPlay, "The strategy should have played at least one card.");
    }

    private static void AssertObeysRules(GameState state, IReadOnlyList<GameOption> options, GameOption pick)
    {
        CombatView combat = state.Combat!;
        PlayerState me = state.Players[0];
        int energy = me.CombatState?.Energy ?? 0;
        int unblocked = combat.Enemies.Sum(e => e.Intents.Where(i => i.Damage is not null)
            .Sum(i => i.Damage!.Value * (i.Hits ?? 1))) - me.Block;

        if (pick.Kind == OptionKind.EndTurn)
        {
            // Rule 5: only end the turn when out of energy and nothing free to play.
            Assert.True(energy <= 0, "Ended the turn while still holding energy.");
            Assert.DoesNotContain(options, o => o.Kind == OptionKind.PlayCard && !o.Card!.CostsX && o.Card!.EnergyCost <= 0);
            return;
        }

        Assert.Equal(OptionKind.PlayCard, pick.Kind);
        CardView card = pick.Card!;
        int block = card.Block ?? 0;

        if (block > 0 && (card.Damage ?? 0) == 0)
        {
            // A pure block card is only played to defend: we must be taking damage and getting ≥80% value.
            Assert.True(unblocked > 0, "Played a block card without any incoming unblocked damage.");
            Assert.True(unblocked * 5 >= block * 4, "Played a block card that would be mostly wasted (<80% used).");
        }
        else if (pick.TargetCombatId is { } targetId)
        {
            // A targeted attack must aim at a lowest-health hittable enemy (ties allowed), unless it is a
            // lethal hit securing a kill on some other target.
            EnemyView target = combat.Enemies.Single(e => e.CombatId == targetId);
            int minHp = combat.Enemies.Where(e => e.IsHittable && e.CurrentHp > 0).Min(e => e.CurrentHp);
            bool lethal = (card.Damage ?? 0) >= target.CurrentHp + target.Block;
            Assert.True(target.CurrentHp == minHp || lethal,
                $"Attacked hp={target.CurrentHp} but a weaker enemy (hp={minHp}) was available and this wasn't a kill.");
        }
    }
}
