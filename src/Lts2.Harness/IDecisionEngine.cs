using System.Collections.Generic;
using System.Linq;

namespace Lts2.Harness;

/// <summary>
/// A single scored recommendation: one legal <see cref="GameOption"/> the engine considered, with a
/// numeric preference (<see cref="Score"/>, higher = more preferred) and an optional human-readable
/// <see cref="Rationale"/>. Scores are only meaningful *relative to each other within one
/// <see cref="IDecisionEngine.Evaluate"/> result* — they are a ranking signal, not an absolute value,
/// and engines are free to use whatever scale they like.
/// </summary>
public sealed record ScoredOption(GameOption Option, double Score, string? Rationale = null);

/// <summary>
/// A pluggable decision policy: given the current observable <see cref="GameState"/> and the legal
/// <see cref="GameOption"/>s for it, produce a score for each option the engine wishes to rank. This is
/// the seam intended for agent training and evaluation — a run driver can swap engines (rules, learned,
/// random, human) behind this one interface.
///
/// <para>The two inputs cover both styles of engine:</para>
/// <list type="bullet">
///   <item>An engine that reasons purely from <paramref name="state"/> (e.g. a learned value/policy net
///   that emits action logits) can produce candidate actions and <b>mask</b> them against
///   <paramref name="options"/> — only options in the supplied list are legal.</item>
///   <item>A simple engine (e.g. <see cref="RulesDecisionEngine"/>) can score the supplied options
///   directly.</item>
/// </list>
///
/// <para>Contract: an engine returns a <see cref="ScoredOption"/> for each option it has an opinion on;
/// it <b>may omit</b> options it does not rank (an omitted option is treated as "no recommendation", not
/// score 0). Returning an empty list means the engine declines to choose for this state (e.g. a
/// combat-only engine off the battlefield). The returned options must be drawn from
/// <paramref name="options"/> — an engine never invents an action. Implementations should be pure over
/// their inputs (no game mutation), so the same state yields the same scores.</para>
/// </summary>
public interface IDecisionEngine
{
    /// <summary>A short stable identifier for the engine, shown in UIs and eval logs (e.g. "Rules").</summary>
    string Name { get; }

    /// <summary>
    /// Score the legal <paramref name="options"/> for <paramref name="state"/>. See the interface remarks
    /// for the contract (subset allowed, options must come from the supplied list, empty = decline).
    /// </summary>
    IReadOnlyList<ScoredOption> Evaluate(GameState state, IReadOnlyList<GameOption> options);
}

/// <summary>Convenience helpers over <see cref="IDecisionEngine"/> for callers that just want the pick.</summary>
public static class DecisionEngineExtensions
{
    /// <summary>
    /// The highest-scored recommendation for <paramref name="state"/>, or null if the engine has no
    /// opinion (an empty <see cref="IDecisionEngine.Evaluate"/> result). Ties are broken toward the option
    /// that appears first in <paramref name="options"/>, so a deterministic engine gives a stable pick.
    /// </summary>
    public static ScoredOption? Best(
        this IDecisionEngine engine, GameState state, IReadOnlyList<GameOption> options)
    {
        ScoredOption? best = null;
        foreach (ScoredOption s in engine.Evaluate(state, options))
        {
            if (best is null || s.Score > best.Score)
            {
                best = s;
            }
        }
        return best;
    }

    /// <summary>The engine's single recommended option (the <see cref="Best"/> option), or null.</summary>
    public static GameOption? Recommend(
        this IDecisionEngine engine, GameState state, IReadOnlyList<GameOption> options) =>
        engine.Best(state, options)?.Option;
}
