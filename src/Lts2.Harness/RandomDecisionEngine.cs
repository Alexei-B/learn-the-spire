using System;
using System.Collections.Generic;

namespace Lts2.Harness;

/// <summary>
/// A baseline <see cref="IDecisionEngine"/> that scores every legal option with a deterministic pseudo-
/// random value. Useful as a control in evaluation (a policy should beat random) and as a smoke-test
/// driver that exercises every phase — unlike <see cref="RulesDecisionEngine"/> it has an opinion on
/// <em>all</em> options, in every phase, which also demonstrates the masking contract: it only ever
/// scores options from the supplied list, so its "recommendation" is always legal.
///
/// <para>Determinism: seeded from a fixed seed by default, so a given engine instance replays the same
/// choices; pass a seed for reproducible eval runs, or reuse one instance to keep advancing the stream.</para>
/// </summary>
public sealed class RandomDecisionEngine : IDecisionEngine
{
    private readonly Random _rng;

    public RandomDecisionEngine(int seed = 1) => _rng = new Random(seed);

    public string Name => "Random";

    public IReadOnlyList<ScoredOption> Evaluate(GameState state, IReadOnlyList<GameOption> options)
    {
        var scored = new List<ScoredOption>(options.Count);
        foreach (GameOption o in options)
        {
            scored.Add(new ScoredOption(o, _rng.NextDouble(), "Random baseline."));
        }
        return scored;
    }
}
