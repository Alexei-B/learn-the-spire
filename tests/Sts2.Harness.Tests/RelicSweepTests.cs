using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Breadth sweep over <see cref="ModelDb.AllRelics"/>: grants every relic in the game and drives a
/// short seeded combat (plus its post-combat rewards) to a terminal state through the public option
/// API. Mirrors <see cref="Act1FightsTests"/>, but varies the relic rather than the encounter, to
/// catch relics whose on-obtain effects, combat-start/turn hooks, on-hit/on-play triggers, or
/// reward-modifying hooks NRE on a null UI singleton (the same class of gap as SoulNexus / KaiserCrab
/// / ScreenShakeTrauma). A relic "works" if the harness grants it (surfacing it in
/// <see cref="PlayerState.Relics"/>) and drives the fight to post-combat rewards without throwing.
/// </summary>
public sealed class RelicSweepTests
{
    private readonly ITestOutputHelper _out;

    public RelicSweepTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Every relic in the game, by id. Booting the runtime (idempotent) is required to enumerate
    /// <see cref="ModelDb"/>; discovery pays that cost once.
    /// </summary>
    public static IEnumerable<object[]> AllRelics
    {
        get
        {
            GameRuntime.EnsureInitialized();
            return ModelDb.AllRelics
                .Select(r => r.Id.Entry)
                .Distinct()
                .OrderBy(id => id, StringComparer.Ordinal)
                .Select(id => new object[] { id })
                .ToList();
        }
    }

    [Theory]
    [MemberData(nameof(AllRelics))]
    public async Task Relic_GrantsAndSurvivesShortCombat(string relicId)
    {
        var t = Task.Run(() => RunWithRelic(relicId));
        await t.WaitAsync(TimeSpan.FromSeconds(90));
    }

    private void RunWithRelic(string relicId)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");

        // Buff to a huge HP pool so the greedy combat survives and we exercise the relic's hooks
        // rather than dying to chip damage.
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999);

        RelicModel relic = ModelDb.AllRelics.First(r => r.Id.Entry == relicId);
        host.ObtainRelicDebug(relic);

        // A relic whose on-obtain effect spawns a reward (e.g. Kaleidoscope's bonus card rewards via
        // OfferCustom) surfaces the reward screen as soon as it is granted; clear it back to the map
        // before entering combat, so the run starts the fight from a stable map phase.
        if (host.GetState().Phase != GamePhase.Map)
        {
            AutoPlayer.Advance(
                host,
                stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
                maxSteps: 200,
                log: _out);
        }

        // The relic should now be on the player.
        Assert.Contains(relicId, host.GetState().Players[0].Relics);

        EncounterModel encounter = Act1FightsTests.ResolveEncounter("SlimesWeak");
        _out.WriteLine($"Entering SlimesWeak with relic {relicId}");
        host.EnterEncounterDebug(encounter);

        Assert.True(host.InCombat, $"expected to be in combat after entering SlimesWeak with {relicId}");

        // Play the fight (and its rewards) out until it reaches a terminal state — back on the map
        // (won) or game over (died). Reaching it without the harness throwing or hanging is the guard.
        GameState end = AutoPlayer.Advance(
            host,
            stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
            maxSteps: 3000,
            log: _out);

        _out.WriteLine($"{relicId} ended: phase={end.Phase} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp}");

        Assert.False(host.InCombat, $"{relicId} combat should have ended");
        Assert.True(
            end.Phase is GamePhase.Map or GamePhase.GameOver,
            $"{relicId} did not resolve to a terminal state; stopped on {end.Phase}");
    }
}
