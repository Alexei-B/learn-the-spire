using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// M8 — seeded property-style end-to-end fuzzing. Each case is a (game-seed, input-seed) pair: the
/// game seed fixes the run's content/RNG, the input seed drives <see cref="RandomPlayer"/>'s random
/// legal choices. The driver plays a full run forward through the public option API, asserting the
/// mechanical invariants (HP/energy/gold/pile/floor sanity) after every step and that the run never
/// gets stuck (a non-terminal state with no legal option) or throws. A failing pair is named in the
/// case data, pinning the regression for replay.
/// </summary>
public sealed class PropertyE2ETests
{
    private readonly ITestOutputHelper _out;

    public PropertyE2ETests(ITestOutputHelper output) => _out = output;

    // The seeded corpus. Each pair plays one full random run; add any pair that surfaces a failure.
    public static IEnumerable<object[]> SeedPairs => new[]
    {
        new object[] { "FUZZ_A", 1 },
        new object[] { "FUZZ_B", 2 },
        new object[] { "FUZZ_C", 3 },
        new object[] { "FUZZ_D", 4 },
        new object[] { "FUZZ_E", 5 },
        new object[] { "FUZZ_F", 6 },
    };

    [Theory]
    [MemberData(nameof(SeedPairs))]
    public async Task RandomLegalPlay_MaintainsInvariants_AndReachesATerminalState(string gameSeed, int inputSeed)
    {
        GameState end = await Task.Run(() => RunOne(gameSeed, inputSeed))
            .WaitAsync(TimeSpan.FromSeconds(180));

        _out.WriteLine($"seed={gameSeed} input={inputSeed} ended phase={end.Phase} floor={end.Floor} " +
                       $"victory={end.IsVictory} hp={end.Players[0].CurrentHp}/{end.Players[0].MaxHp}");

        // Random play almost always dies eventually; reaching a terminal game-over (or, rarely, a
        // victory) within the step budget proves the run drove to completion without getting stuck.
        Assert.True(end.Phase == GamePhase.GameOver,
            $"random run did not reach a terminal state (stopped at {end.Phase}, floor {end.Floor})");
    }

    private GameState RunOne(string gameSeed, int inputSeed)
    {
        using var errors = new LogErrorSink();
        GameHost host = TestNav.StartOnMap(gameSeed);
        GameState end = RandomPlayer.PlayFullRun(host, inputSeed, maxSteps: 6000, log: _out);
        AssertNoSwallowedErrors(errors);
        return end;
    }

    /// <summary>
    /// Fail if the run logged any error-level message — these are exceptions the game swallowed on
    /// fire-and-forget tasks (e.g. an NRE on a null UI singleton inside an event option), which would
    /// otherwise leave the run looking healthy. Turns those invisible faults into a test failure.
    /// </summary>
    private void AssertNoSwallowedErrors(LogErrorSink errors)
    {
        IReadOnlyList<string> captured = errors.Errors;
        foreach (string e in captured)
        {
            _out.WriteLine($"SWALLOWED ERROR: {e}");
        }
        Assert.True(captured.Count == 0,
            $"the run swallowed {captured.Count} error-level log(s); first: {(captured.Count > 0 ? captured[0] : "")}");
    }

    [Theory]
    [MemberData(nameof(SeedPairs))]
    public async Task RandomLegalPlay_Buffed_ExercisesDeeperContent_MaintainingInvariants(string gameSeed, int inputSeed)
    {
        // Buff HP so the random run survives early combats and reaches more room types (events,
        // treasure, shops, rest sites, act transitions) — broadening invariant coverage. The driver
        // checks invariants after every step and surfaces any stuck state; we additionally require it
        // to climb past the opening floors (proving it drove through several rooms) and terminate.
        GameState end = await Task.Run(() => RunBuffed(gameSeed, inputSeed))
            .WaitAsync(TimeSpan.FromSeconds(240));

        _out.WriteLine($"[buffed] seed={gameSeed} input={inputSeed} ended phase={end.Phase} floor={end.Floor} " +
                       $"victory={end.IsVictory}");

        Assert.Equal(GamePhase.GameOver, end.Phase);
        Assert.True(end.Floor >= 4, $"buffed random run barely advanced (floor {end.Floor})");
    }

    private GameState RunBuffed(string gameSeed, int inputSeed)
    {
        using var errors = new LogErrorSink();
        GameHost host = TestNav.StartOnMap(gameSeed);
        TestNav.SetHp(host, maxHp: 140, currentHp: 140);
        GameState end = RandomPlayer.PlayFullRun(host, inputSeed, maxSteps: 8000, log: _out);
        AssertNoSwallowedErrors(errors);
        return end;
    }
}
