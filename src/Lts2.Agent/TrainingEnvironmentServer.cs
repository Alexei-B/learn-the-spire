using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Lts2.Agent.Wire;
using Lts2.Harness;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Lts2.Agent;

/// <summary>
/// A headless "gym"-style environment server: it drives one <see cref="GameHost"/> from
/// <c>reset</c>/<c>step</c>/<c>close</c> commands read off an <see cref="ILineChannel"/>, replying to
/// each with an <see cref="Observation"/> (the full state + legal options + terminal flag + reward
/// scalars). A remote training loop (e.g. a Python agent) is the driver; this process is the
/// environment. The action encoding is identical to what the evaluation <see cref="ProcessDecisionEngine"/>
/// sends, so a policy trained here plugs straight into the TUI.
///
/// <para><b>One run per process.</b> The game keeps run/combat state in process-wide singletons, so a
/// single server hosts a single run at a time (<c>reset</c> tears down and restarts it via
/// <see cref="GameHost.StartNewRun"/>). A vectorized trainer runs N of these processes in parallel.</para>
/// </summary>
public sealed class TrainingEnvironmentServer
{
    private GameHost? _host;
    // Non-null while serving a combat scenario (reset_combat); null in full-run mode (reset).
    private CombatScenario.Spec? _scenario;
    // The options from the last observation the driver received. A step's index refers to exactly this
    // list, and the state cannot change between that Observe and the next Step, so we reuse it for the
    // step's validation/apply instead of recomputing ListOptions (which is ~1/4 of the per-step C# cost).
    private IReadOnlyList<GameOption>? _lastOptions;

    /// <summary>
    /// Serve commands from <paramref name="channel"/> until it hits end-of-stream or a <c>close</c>
    /// command. Boots the game runtime up front so the first <c>reset</c> is fast and any load error
    /// surfaces immediately.
    /// </summary>
    public void Serve(ILineChannel channel)
    {
        if (channel is null)
        {
            throw new ArgumentNullException(nameof(channel));
        }

        GameRuntime.EnsureInitialized();

        string? line;
        while ((line = channel.ReadLine()) is not null)
        {
            if (line.Length == 0)
            {
                continue;
            }

            (string response, bool close) = Handle(line);
            channel.WriteLine(response);
            if (close)
            {
                break;
            }
        }
    }

    private (string Response, bool Close) Handle(string line)
    {
        EnvCommand command;
        try
        {
            command = JsonSerializer.Deserialize<EnvCommand>(line, AgentJson.Options)
                ?? throw new InvalidOperationException("Empty command.");
        }
        catch (Exception ex)
        {
            return (Error($"Could not parse command: {ex.Message}"), false);
        }

        try
        {
            switch (command.Cmd)
            {
                case "reset":
                    return (Reset(command), false);
                case "reset_combat":
                    return (ResetCombat(command), false);
                case "step":
                    return (Step(command), false);
                case "close":
                    return (JsonSerializer.Serialize(new OkResponse(), AgentJson.Options), true);
                default:
                    return (Error($"Unknown command '{command.Cmd}'."), false);
            }
        }
        catch (Exception ex)
        {
            // Full exception (message + stack) to stderr for debugging (suppressed unless the driver
            // sets log_stderr); the wire error stays the concise message.
            Console.Error.WriteLine($"[env-error] {ex}");
            return (Error(ex.Message), false);
        }
    }

    private string Reset(EnvCommand command)
    {
        _scenario = null; // full-run mode
        string seed = string.IsNullOrEmpty(command.Seed) ? "AGENT" : command.Seed!;
        int ascension = command.Ascension ?? 0;
        CharacterModel character = ResolveCharacter(command.Character);

        _host = GameHost.StartNewRun(seed, new[] { character }, ascension);
        _host.EnterFirstRoom();
        return Observe();
    }

    private string ResetCombat(EnvCommand command)
    {
        string seed = string.IsNullOrEmpty(command.Seed) ? "AGENT" : command.Seed!;
        CombatScenario.DeckSpec? deckSpec = ParseDeckSpec(command.DeckSpec);

        // Explicit deck (closed evals): the deckSpec "explicit" form, or the legacy top-level `cards`.
        IReadOnlyList<string>? explicitCards =
            (deckSpec as CombatScenario.DeckSpec.Explicit)?.CardIds
            ?? (command.Cards is { Count: > 0 } ? command.Cards : null);
        if (explicitCards is { Count: > 0 })
        {
            // Fully-specified closed-eval scenario: exact character + deck + encounter (always full build).
            string character = string.IsNullOrEmpty(command.Character) ? "IRONCLAD" : command.Character!;
            string encounter = command.Encounter
                ?? throw new InvalidOperationException("reset_combat with an explicit deck also needs an 'encounter'.");
            (GameHost host, CombatScenario.Spec spec) = CombatScenario.CreateExplicit(
                seed, character, explicitCards, command.Relics, encounter, command.EnemyHp);
            _host = host;
            _scenario = spec with { DeckKind = "explicit" };
            return Observe();
        }

        var rng = new Random(StableSeed(seed));
        double elite = command.ElitePct ?? 0.2, boss = command.BossPct ?? 0.05;
        bool starter = command.StarterDeck ?? false;
        int act = command.Act ?? -1;

        // Soft reset (~80x cheaper than a fresh StartNewRun): reuse the live run when the requested
        // character matches the one it was built for. A run is a single character, so a caller keeps an
        // env pinned to one character and gets character diversity by spreading them across env processes.
        if (_host is { } live && CanSoftReset(live, command.Character) && live.IsReadyForSoftReenter)
        {
            _scenario = CombatScenario.Reenter(live, rng, elite, boss, starter, act, deckSpec);
            return Observe();
        }

        (GameHost created, CombatScenario.Spec createdSpec) = CombatScenario.Create(
            seed, rng, command.Character, elite, boss, starter, act, deckSpec);
        _host = created;
        _scenario = createdSpec;
        return Observe();
    }

    /// <summary>Parse the optional <c>deckSpec</c> wire object into a <see cref="CombatScenario.DeckSpec"/>.
    /// The <c>cards</c> field is an int for <c>"random"</c> but an array for <c>"explicit"</c>, so it is read
    /// element-by-element rather than bound to a typed property. All realistic defaults match the roadmap
    /// contract, so <c>{"kind":"realistic"}</c> alone is valid.</summary>
    private static CombatScenario.DeckSpec? ParseDeckSpec(JsonElement? element)
    {
        if (element is not { } e || e.ValueKind != JsonValueKind.Object)
        {
            return null;
        }
        string kind = e.TryGetProperty("kind", out JsonElement k) ? k.GetString() ?? "" : "";
        switch (kind.ToLowerInvariant())
        {
            case "random":
            {
                int cards = e.TryGetProperty("cards", out JsonElement c) && c.ValueKind == JsonValueKind.Number
                    ? c.GetInt32() : 15;
                return new CombatScenario.DeckSpec.Random(cards);
            }
            case "realistic":
            {
                CombatScenario.DeckSpec.Realistic d = CombatScenario.DeckSpec.DefaultRealistic();
                (int rMin, int rMax) = ReadRange(e, "removals", d.RemovalsMin, d.RemovalsMax);
                (int aMin, int aMax) = ReadRange(e, "additions", d.AdditionsMin, d.AdditionsMax);
                CombatScenario.PoolWeights w = ReadWeights(e, d.Weights);
                return new CombatScenario.DeckSpec.Realistic(rMin, rMax, aMin, aMax, w);
            }
            case "explicit":
            {
                var ids = new List<string>();
                if (e.TryGetProperty("cards", out JsonElement arr) && arr.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement item in arr.EnumerateArray())
                    {
                        if (item.GetString() is { } id)
                        {
                            ids.Add(id);
                        }
                    }
                }
                return new CombatScenario.DeckSpec.Explicit(ids);
            }
            default:
                throw new InvalidOperationException(
                    $"Unknown deckSpec kind '{kind}' (expected 'random', 'realistic', or 'explicit').");
        }
    }

    /// <summary>Read a two-element inclusive <c>[min,max]</c> range field, falling back to the defaults.</summary>
    private static (int Min, int Max) ReadRange(JsonElement obj, string name, int defMin, int defMax)
    {
        if (obj.TryGetProperty(name, out JsonElement arr) && arr.ValueKind == JsonValueKind.Array
            && arr.GetArrayLength() >= 2)
        {
            int min = arr[0].GetInt32();
            int max = arr[1].GetInt32();
            return (min, max);
        }
        return (defMin, defMax);
    }

    /// <summary>Read the addition pool weights, falling back per-key to the defaults.</summary>
    private static CombatScenario.PoolWeights ReadWeights(JsonElement obj, CombatScenario.PoolWeights def)
    {
        if (!obj.TryGetProperty("weights", out JsonElement w) || w.ValueKind != JsonValueKind.Object)
        {
            return def;
        }
        double Read(string name, double fallback) =>
            w.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;
        return new CombatScenario.PoolWeights(
            Read("own", def.Own), Read("colorless", def.Colorless),
            Read("curse", def.Curse), Read("offCharacter", def.OffCharacter));
    }

    /// <summary>Whether the live run can be soft-reset into a new fight for <paramref name="wanted"/>.
    /// A run is one character: when a specific character is requested we reuse the run only if it matches;
    /// when none is requested (random-character training) we <em>pin</em> to whatever character this env's
    /// run was first built with, so it can still soft-reset (diversity then comes from spreading characters
    /// across env processes, whose differing seeds pick different first characters).</summary>
    // Soft reset makes a fresh fight ~80x cheaper (reuse the run + EnterEncounterDebug vs a full
    // StartNewRun) and is correct for a handful of fights, but reusing one run across MANY fights leaks
    // combat state unboundedly: EnterEncounterDebug starts a new combat without the game's real combat-end
    // / room-transition teardown, so cards, piles, NetCombatCardDb registrations and (worst) millions of
    // event-subscription delegates accumulate (2.5 GB+ within one training iteration → GC thrash that
    // stalls the process). Patching individual leak sources (CombatManager.Reset, NetCombatCardDb clear,
    // map-history — see GameHost.PrepareForSoftReenter) is not enough; the delegate leak has too many
    // channels. So it is OFF by default and not currently viable for real training — the iteration-speed
    // win comes from the stable full-reset path + more env processes. Opt in with LTS2_SOFT_RESET=1 only
    // for short isolated/benchmarking use.
    private static readonly bool SoftResetEnabled =
        Environment.GetEnvironmentVariable("LTS2_SOFT_RESET") is "1" or "true";

    private static bool CanSoftReset(GameHost host, string? wanted)
    {
        if (!SoftResetEnabled)
        {
            return false;
        }
        try
        {
            if (!RunManager.Instance.IsInProgress || host.Run.IsGameOver)
            {
                // A game-over run (the player died in the last fight) poisons the next combat's turn
                // signalling — rebuild it fresh. (Leftover action/choice residue is handled separately by
                // TryClearCombatResidue, which falls back to a full reset if it can't be cleared.)
                return false;
            }
            return string.IsNullOrEmpty(wanted)
                || host.Run.Players[0].Character.Id.Entry.Contains(wanted!, StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>A process-stable hash of the seed (unlike string.GetHashCode, which is randomized per
    /// process) so a given seed reproduces the same random scenario composition.</summary>
    private static int StableSeed(string s)
    {
        unchecked
        {
            int h = 17;
            foreach (char c in s)
            {
                h = h * 31 + c;
            }
            return h;
        }
    }

    private string Step(EnvCommand command)
    {
        GameHost host = _host
            ?? throw new InvalidOperationException("No run in progress; send a 'reset' before 'step'.");

        if (command.CardIndices is { } cardIndices)
        {
            // A "choose N of M" card choice: resolve any valid subset (single-index picks are also
            // enumerated as options, but the combinatorial case needs the explicit index list).
            host.ApplyCardChoice(cardIndices);
        }
        else if (command.Index is { } index)
        {
            long t = System.Diagnostics.Stopwatch.GetTimestamp();
            // The index refers to the options from the last observation; reuse them (state is unchanged
            // since then) rather than recomputing.
            IReadOnlyList<GameOption> options = _lastOptions ?? host.ListOptions();
            t = StepProfiler.Mark("listOptions_validate", t);
            if (index < 0 || index >= options.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(command.Index), index, $"Action index out of range (expected 0..{options.Count - 1}).");
            }
            host.Apply(options[index]);
            StepProfiler.Mark("apply", t);
        }
        else
        {
            throw new InvalidOperationException("A 'step' needs an 'index' or 'cardIndices'.");
        }

        string obs = Observe();
        StepProfiler.StepDone();
        return obs;
    }

    private string Observe()
    {
        GameHost host = _host!;

        long t = System.Diagnostics.Stopwatch.GetTimestamp();
        GameState state = host.GetState();
        t = StepProfiler.Mark("getState", t);
        IReadOnlyList<GameOption> options = host.ListOptions();
        _lastOptions = options;   // reused by the next Step's validation/apply (state is unchanged until then)
        t = StepProfiler.Mark("listOptions", t);
        Observation obs = _scenario is { } spec
            ? BuildScenarioObservation(host, state, options, spec)
            : Observation.From(state, options);
        t = StepProfiler.Mark("buildObs", t);
        string json = JsonSerializer.Serialize(obs, AgentJson.Options);
        StepProfiler.Mark("serialize", t);
        return json;
    }

    /// <summary>
    /// Build a scenario observation: the episode is a single fight, so <c>Done</c> marks the combat
    /// ending (won or lost), and the info carries the outcome + HP lost during the fight (with the
    /// character's end-of-combat starter heal added back on a win — see <see cref="CombatScenario"/>).
    /// </summary>
    private static Observation BuildScenarioObservation(
        GameHost host, GameState state, IReadOnlyList<GameOption> options, CombatScenario.Spec spec)
    {
        // The fight is over once no combat move is on offer. Keying off the actual options is the
        // reliable signal: host.InCombat can linger true on the post-combat reward screen, and the
        // phase can momentarily read Combat while rewards are already being offered — either way, if
        // the only options are rewards/map (no PlayCard/EndTurn/potion/mid-combat choice), the fight
        // has ended and the agent must not act on those.
        bool canStillFight = host.InCombat && options.Any(o =>
            o.Kind is OptionKind.PlayCard or OptionKind.EndTurn
                or OptionKind.UsePotion or OptionKind.DiscardPotion or OptionKind.SelectCards);
        bool combatOver = !canStillFight;
        bool playerAlive = host.Run.Players[0].Creature.IsAlive;
        bool won = combatOver && playerAlive;
        int? hpLost = null;
        if (combatOver)
        {
            int endHp = state.Players.Count > 0 ? state.Players[0].CurrentHp : 0;
            int loss = spec.StartHp - endHp + (won ? spec.StarterHeal : 0);
            hpLost = Math.Clamp(loss, 0, spec.StartHp);
        }

        var players = new List<PlayerInfo>(state.Players.Count);
        foreach (PlayerState p in state.Players)
        {
            players.Add(new PlayerInfo { CurrentHp = p.CurrentHp, MaxHp = p.MaxHp, Gold = p.Gold });
        }

        return new Observation
        {
            State = state,
            Options = options,
            Done = combatOver,
            Info = new ObservationInfo
            {
                Score = state.Score,
                Phase = state.Phase.ToString(),
                Floor = state.Floor,
                Act = spec.Act,
                GameOver = state.IsGameOver,
                Victory = won,
                Players = players,
                CombatOver = combatOver,
                Won = won,
                HpLost = hpLost,
                Encounter = spec.Encounter,
                RoomType = spec.RoomType,
                DeckSpec = spec.DeckKind,
                RemovedCards = spec.RemovedCards,
                AddedCards = spec.AddedCards,
            },
        };
    }

    private static CharacterModel ResolveCharacter(string? name) =>
        string.IsNullOrEmpty(name)
            ? ModelDb.AllCharacters.First()
            : ModelDb.AllCharacters.First(
                c => c.Id.Entry.Contains(name!, StringComparison.OrdinalIgnoreCase));

    private static string Error(string message) =>
        JsonSerializer.Serialize(new ErrorResponse { Error = message }, AgentJson.Options);
}

/// <summary>
/// Opt-in per-step timing (set <c>LTS2_PROFILE=1</c>) that accumulates how long each part of a step
/// takes — <c>listOptions</c>, <c>getState</c> (projection), <c>buildObs</c>, <c>serialize</c>,
/// <c>apply</c> (the game logic + pump) — and prints an average µs/step breakdown to stderr every N
/// steps. Zero cost when disabled. Lets us optimize the measured hotspot instead of guessing.
/// </summary>
internal static class StepProfiler
{
    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("LTS2_PROFILE") is "1" or "true";
    private const int ReportEvery = 500;

    private static readonly object Lock = new();
    private static readonly Dictionary<string, long> Ticks = new();
    private static int _steps;

    /// <summary>Record the time since <paramref name="since"/> under <paramref name="category"/> and
    /// return a fresh timestamp (so calls chain: <c>t = Mark("a", t); t = Mark("b", t);</c>).</summary>
    public static long Mark(string category, long since)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (Enabled)
        {
            lock (Lock)
            {
                Ticks.TryGetValue(category, out long acc);
                Ticks[category] = acc + (now - since);
            }
        }
        return now;
    }

    public static void StepDone()
    {
        if (!Enabled)
        {
            return;
        }
        lock (Lock)
        {
            if (++_steps % ReportEvery != 0)
            {
                return;
            }
            double usPerStep = 1_000_000.0 / System.Diagnostics.Stopwatch.Frequency / _steps;
            string parts = string.Join(" ", Ticks.OrderByDescending(kv => kv.Value)
                .Select(kv => $"{kv.Key}={kv.Value * usPerStep:0}us"));
            Console.Error.WriteLine($"[profile] steps={_steps} per-step: {parts}");
        }
    }
}
