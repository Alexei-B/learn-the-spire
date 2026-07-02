using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Encounters;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.ValueProps;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// The Decimillipede elite is three segments that revive each other until all are dead at once. This
/// checks that killing every segment actually ends the fight (headless it must not depend on the
/// UI-driven fade-out to remove the dead segments).
/// </summary>
public sealed class DecimillipedeTests
{
    private readonly ITestOutputHelper _out;

    public DecimillipedeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task KillingAllSegments_EndsTheFight()
    {
        await Task.Run(Run).WaitAsync(TimeSpan.FromSeconds(60));
    }

    private void Run()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        host.EnterEncounterDebug(ModelDb.Encounter<DecimillipedeElite>());
        Assert.True(host.InCombat, "expected to be fighting the Decimillipede");

        CombatState combat = host.Combat!;
        var segments = combat.Enemies.ToList();
        _out.WriteLine($"segments: {segments.Count} hp=[{string.Join(",", segments.Select(s => s.CurrentHp))}]");
        Assert.Equal(3, segments.Count);

        var ctx = new ThrowingPlayerChoiceContext();
        foreach (Creature seg in segments)
        {
            CreatureCmd.Damage(ctx, seg, 9999m, ValueProp.Unblockable | ValueProp.Unpowered, dealer: null, cardSource: null)
                .GetAwaiter().GetResult();
        }
        RunManager.Instance.ActionExecutor.FinishedExecutingActions().GetAwaiter().GetResult();

        // The combat flow checks the win condition after each action (card play / enemy turn); the
        // direct-damage seam above doesn't, so trigger it the way real play would.
        _out.WriteLine($"pre-check: inCombat={host.InCombat} isEnding={CombatManager.Instance.IsEnding} " +
            $"alive={segments.Count(s => s.IsAlive)}");
        CombatManager.Instance.CheckWinCondition().GetAwaiter().GetResult();
        RunManager.Instance.ActionExecutor.FinishedExecutingActions().GetAwaiter().GetResult();

        _out.WriteLine($"after: inCombat={host.InCombat} alive={segments.Count(s => s.IsAlive)} " +
            $"dead=[{string.Join(",", segments.Select(s => s.IsDead))}] enemiesInState={combat.Enemies.Count}");

        Assert.All(segments, s => Assert.True(s.IsDead, "every segment should be dead"));
        Assert.False(host.InCombat, "the fight should end once all segments are dead");
    }
}
