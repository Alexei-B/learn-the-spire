using System.Collections.Generic;
using System.Linq;
using Lts2.Harness;
using Xunit;

namespace Lts2.Harness.Tests;

/// <summary>
/// Contract tests for the <see cref="IDecisionEngine"/> seam (state + options in, scored actions out) —
/// the interface agent training/evaluation and the TUI's auto-play both consume. Covers the two shipped
/// engines: <see cref="RulesDecisionEngine"/> (combat-only) and <see cref="RandomDecisionEngine"/>
/// (all-phase baseline), asserting the scored list is legal (a subset of the supplied options), that
/// <c>Best</c>/<c>Recommend</c> agree with the top score, and the masking/decline contract.
/// </summary>
public sealed class DecisionEngineTests
{
    [Fact]
    public void RulesEngine_InCombat_ScoresListedOptions_AndBestMatchesTopScore()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        var engine = new RulesDecisionEngine();

        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();
        IReadOnlyList<ScoredOption> scored = engine.Evaluate(state, options);

        Assert.NotEmpty(scored);
        // Every scored option is one of the legal options — an engine never invents a move.
        Assert.All(scored, s => Assert.Contains(s.Option, options));
        // No duplicates: each option is scored at most once.
        Assert.Equal(scored.Select(s => s.Option).Distinct().Count(), scored.Count);

        // Best / Recommend agree with the highest score, breaking ties toward the earliest option.
        ScoredOption expectedBest = scored.Aggregate((a, b) => b.Score > a.Score ? b : a);
        Assert.Same(expectedBest.Option, engine.Best(state, options)!.Option);
        Assert.Same(expectedBest.Option, engine.Recommend(state, options));
    }

    [Fact]
    public void RulesEngine_OutOfCombat_Declines()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        var engine = new RulesDecisionEngine();

        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();
        Assert.NotEqual(GamePhase.Combat, state.Phase);
        Assert.NotEmpty(options); // there are moves to make…

        Assert.Empty(engine.Evaluate(state, options)); // …but the combat-only policy has no opinion.
        Assert.Null(engine.Recommend(state, options));
    }

    [Fact]
    public void RandomEngine_ScoresEveryOption_AndAlwaysRecommendsALegalMove_EvenOutOfCombat()
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        // A fixed seed makes the baseline deterministic (reproducible eval runs).
        var engine = new RandomDecisionEngine(seed: 42);

        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();
        IReadOnlyList<ScoredOption> scored = engine.Evaluate(state, options);

        // Masking contract: it scores exactly the supplied options, so its pick is always legal.
        Assert.Equal(options.Count, scored.Count);
        Assert.All(scored, s => Assert.Contains(s.Option, options));
        Assert.Contains(engine.Recommend(state, options), options);
    }
}
