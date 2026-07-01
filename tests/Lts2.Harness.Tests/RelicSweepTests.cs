using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

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

        RelicModel relic = ModelDb.AllRelics.First(r => r.Id.Entry == relicId).ToMutable();

        // Skip relics the game would never grant in this single-player run: it filters relic pools by
        // IsAllowed, so granting a disallowed one exercises an impossible state. e.g. MassiveScroll is
        // multiplayer-only (IsAllowed => Players.Count > 1); in single-player its card pool is empty
        // and its on-obtain card generation throws.
        if (!relic.IsAllowed(host.Run))
        {
            _out.WriteLine($"Skipping {relicId}: not allowed in a single-player run.");
            return;
        }

        // SeaGlass is a character-specific event relic, granted in-game only via the Orobas event,
        // which assigns its CharacterId first; granted raw it logs an error and defaults to Ironclad.
        // Mirror Orobas: pin it to the owner's character so it draws its cards from a real pool.
        if (relic is MegaCrit.Sts2.Core.Models.Relics.SeaGlass seaGlass)
        {
            seaGlass.CharacterId = host.Run.Players[0].Character.Id;
        }

        // Some event relics need per-player setup before they're granted — their event calls
        // SetupForPlayer first to pick a card/relic from the player's pool that AfterObtained then
        // uses (DustyTome/ArchaicTooth/TouchOfOrobas). Mirror that. A bool-returning setup that
        // fails means the relic isn't applicable to this player (the event skips it), so skip too.
        System.Reflection.MethodInfo? setup = relic.GetType().GetMethod("SetupForPlayer");
        if (setup is not null && setup.Invoke(relic, new object[] { host.Run.Players[0] }) is false)
        {
            _out.WriteLine($"Skipping {relicId}: SetupForPlayer reported it is not applicable.");
            return;
        }

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
