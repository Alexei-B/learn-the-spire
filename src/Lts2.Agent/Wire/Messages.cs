using System.Collections.Generic;
using System.Text.Json;
using Lts2.Harness;

namespace Lts2.Agent.Wire;

// The wire messages. Both protocols share one observation/action encoding:
//   * an OBSERVATION is the immutable GameState serialized as-is plus the legal options, so a policy
//     trained against the training-environment server works unchanged behind the evaluation
//     decision-server — same features, same action ids.
//   * an ACTION identifies a choice by its INDEX into the option list (its array position), or, for a
//     "choose N of M" card choice, by explicit CardIndices (see TrainingEnvironmentServer / the plan's
//     multi-select note).
// GameState and GameOption are serialized directly (their public getters are exactly the observable
// surface; GameOption's live game refs are internal and never serialize). None of these are ever
// deserialized back into game objects — the C# side always maps an index onto the options it just
// listed.

/// <summary>
/// A decision request sent by the evaluation client (the TUI's <c>ProcessDecisionEngine</c>) to an
/// external policy server: "given this state and these legal options, score them". Mirrors
/// <see cref="IDecisionEngine.Evaluate"/> over the wire.
/// </summary>
public sealed record EvaluateRequest
{
    public string Type { get; init; } = "evaluate";
    public int ProtocolVersion { get; init; } = AgentJson.ProtocolVersion;
    public required GameState State { get; init; }
    public required IReadOnlyList<GameOption> Options { get; init; }
}

/// <summary>A policy server's reply to an <see cref="EvaluateRequest"/>: a (possibly empty) subset of
/// scored options. An empty list means the policy declines (mapped to an empty
/// <see cref="IDecisionEngine.Evaluate"/> result — "no recommendation").</summary>
public sealed record ScoresResponse
{
    public IReadOnlyList<ScoreDto>? Scores { get; init; }
}

/// <summary>One scored option in a <see cref="ScoresResponse"/>: <see cref="Index"/> is the option's
/// position in the request's option list.</summary>
public sealed record ScoreDto
{
    public required int Index { get; init; }
    public double Score { get; init; }
    public string? Rationale { get; init; }
}

/// <summary>
/// A command from the training driver (Python) to the environment server: <c>reset</c> a fresh run,
/// <c>step</c> an action, or <c>close</c> the server. Unused fields are null for a given command.
/// </summary>
public sealed record EnvCommand
{
    public required string Cmd { get; init; }

    // reset
    public string? Seed { get; init; }
    public string? Character { get; init; }
    public int? Ascension { get; init; }

    // reset_combat: an isolated randomized fight (see CombatScenario). Character null = random.
    public double? ElitePct { get; init; }
    public double? BossPct { get; init; }
    public bool? StarterDeck { get; init; }   // use the character's fixed starting deck + starter relic
    public int? Act { get; init; }            // restrict a random encounter to this act (0/1/2)

    // reset_combat: an optional deck spec selecting how the deck is built (see CombatScenario.DeckSpec):
    //   {"kind":"random","cards":15} | {"kind":"realistic",...} | {"kind":"explicit","cards":[...]}.
    // Left as a raw element and parsed by the server (the `cards` field is an int for "random" but an
    // array for "explicit", so it can't bind to one typed property). Absent = today's behavior.
    public JsonElement? DeckSpec { get; init; }

    // reset_combat, explicit form (closed evals): an exact deck (card ids) + named encounter, and
    // optionally extra relics + per-enemy HP overrides (for unambiguous situations like a free lethal).
    // When Cards is set the scenario is fully specified and deterministic.
    public IReadOnlyList<string>? Cards { get; init; }
    public IReadOnlyList<string>? Relics { get; init; }
    public string? Encounter { get; init; }
    public IReadOnlyList<int>? EnemyHp { get; init; }

    // step: pick Options[Index], or resolve a card choice with CardIndices (choose N of M).
    public int? Index { get; init; }
    public IReadOnlyList<int>? CardIndices { get; init; }
}

/// <summary>
/// The environment server's reply to <c>reset</c>/<c>step</c>: the full <see cref="GameState"/> plus
/// the legal <see cref="Options"/> for it, a terminal flag, and a compact <see cref="Info"/> block of
/// the scalars a reward function usually wants (so Python need not walk the whole state). Reward is
/// deliberately not computed here — the training loop derives whatever signal it wants from
/// <see cref="Info"/>/<see cref="State"/>.
/// </summary>
public sealed record Observation
{
    public int ProtocolVersion { get; init; } = AgentJson.ProtocolVersion;
    public required GameState State { get; init; }
    public required IReadOnlyList<GameOption> Options { get; init; }
    public required bool Done { get; init; }
    public required ObservationInfo Info { get; init; }

    /// <summary>Project a fresh observation from the current run state.</summary>
    public static Observation From(GameState state, IReadOnlyList<GameOption> options)
    {
        var players = new List<PlayerInfo>(state.Players.Count);
        foreach (PlayerState p in state.Players)
        {
            players.Add(new PlayerInfo
            {
                CurrentHp = p.CurrentHp,
                MaxHp = p.MaxHp,
                Gold = p.Gold,
            });
        }

        return new Observation
        {
            State = state,
            Options = options,
            Done = state.IsGameOver,
            Info = new ObservationInfo
            {
                Score = state.Score,
                Phase = state.Phase.ToString(),
                Floor = state.Floor,
                Act = state.ActIndex,
                GameOver = state.IsGameOver,
                Victory = state.IsVictory,
                Players = players,
            },
        };
    }
}

/// <summary>The reward-relevant scalars, surfaced for convenience (all also derivable from
/// <see cref="Observation.State"/>).</summary>
public sealed record ObservationInfo
{
    public required int Score { get; init; }
    public required string Phase { get; init; }
    public required int Floor { get; init; }
    public required int Act { get; init; }
    public required bool GameOver { get; init; }
    public required bool Victory { get; init; }
    public required IReadOnlyList<PlayerInfo> Players { get; init; }

    // Combat-scenario mode (reset_combat) only; null in full-run mode. The scenario episode is one
    // fight: CombatOver marks the terminal step, Won is the fight outcome, and HpLost is the HP lost
    // during the fight (with the character's end-of-combat starter heal added back — see CombatScenario).
    public bool? CombatOver { get; init; }
    public bool? Won { get; init; }
    public int? HpLost { get; init; }
    public string? Encounter { get; init; }
    public string? RoomType { get; init; }

    // Scenario deck metadata (reset_combat only), so collectors can tag outcomes and report deck
    // distributions without re-deriving them. DeckSpec is the resolved kind ("random"/"realistic"/
    // "explicit"/"starter"); RemovedCards/AddedCards are the realistic sampler's actual picks (card ids),
    // null for other deck kinds.
    public string? DeckSpec { get; init; }
    public IReadOnlyList<string>? RemovedCards { get; init; }
    public IReadOnlyList<string>? AddedCards { get; init; }
}

/// <summary>Per-player reward scalars in an <see cref="ObservationInfo"/>.</summary>
public sealed record PlayerInfo
{
    public required int CurrentHp { get; init; }
    public required int MaxHp { get; init; }
    public required int Gold { get; init; }
}

/// <summary>A generic acknowledgement (e.g. the reply to <c>close</c>).</summary>
public sealed record OkResponse
{
    public bool Ok { get; init; } = true;
}

/// <summary>An error reply for a malformed or failed command/request.</summary>
public sealed record ErrorResponse
{
    public required string Error { get; init; }
}
