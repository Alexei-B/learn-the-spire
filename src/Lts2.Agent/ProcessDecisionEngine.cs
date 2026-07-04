using System;
using System.Collections.Generic;
using System.Text.Json;
using Lts2.Agent.Wire;
using Lts2.Harness;

namespace Lts2.Agent;

/// <summary>
/// An <see cref="IDecisionEngine"/> whose decisions come from an external process over the line
/// protocol — the evaluation-side counterpart to the training environment. Each
/// <see cref="Evaluate"/> serializes an <see cref="EvaluateRequest"/> (the state + legal options),
/// sends it to the managed <see cref="DecisionEngineServerProcess"/>, and maps the returned
/// <see cref="ScoresResponse"/> back onto the supplied options by index. This lets a policy trained
/// in another language (e.g. a Python model) be dropped into the TUI's engine menu.
///
/// <para>Every failure mode — a dead/timed-out process, a malformed reply, an out-of-range index —
/// degrades to an empty result (the seam's "decline" / no-recommendation), logged but never thrown,
/// so a broken agent shows no auto-play pick instead of crashing the caller.</para>
/// </summary>
public sealed class ProcessDecisionEngine : IDecisionEngine, IDisposable
{
    private readonly IDecisionChannel _channel;
    private readonly Action<string>? _log;

    public string Name { get; }

    public ProcessDecisionEngine(string name, IDecisionChannel channel, Action<string>? log = null)
    {
        Name = string.IsNullOrWhiteSpace(name) ? "External" : name;
        _channel = channel ?? throw new ArgumentNullException(nameof(channel));
        _log = log;
    }

    /// <summary>
    /// Launch <paramref name="command"/> as the policy server and wrap it as an engine named
    /// <paramref name="name"/> (shown in the engine menu / eval logs).
    /// </summary>
    public static ProcessDecisionEngine Launch(
        string name,
        string command,
        string? arguments = null,
        string? workingDirectory = null,
        TimeSpan? timeout = null,
        Action<string>? log = null,
        System.Collections.Generic.IReadOnlyDictionary<string, string>? environment = null)
    {
        DecisionEngineServerProcess server =
            DecisionEngineServerProcess.Start(command, arguments, workingDirectory, timeout, log, environment);
        return new ProcessDecisionEngine(name, server, log);
    }

    public IReadOnlyList<ScoredOption> Evaluate(GameState state, IReadOnlyList<GameOption> options)
    {
        if (options.Count == 0)
        {
            return Array.Empty<ScoredOption>();
        }

        try
        {
            var request = new EvaluateRequest { State = state, Options = options };
            string requestLine = JsonSerializer.Serialize(request, AgentJson.Options);

            string? responseLine = _channel.SendRequest(requestLine);
            if (responseLine is null)
            {
                return Array.Empty<ScoredOption>();
            }

            ScoresResponse? response = JsonSerializer.Deserialize<ScoresResponse>(responseLine, AgentJson.Options);
            if (response?.Scores is not { } scores)
            {
                return Array.Empty<ScoredOption>();
            }

            var result = new List<ScoredOption>(scores.Count);
            foreach (ScoreDto s in scores)
            {
                if (s.Index < 0 || s.Index >= options.Count)
                {
                    _log?.Invoke($"[agent] {Name}: dropping out-of-range option index {s.Index} (of {options.Count}).");
                    continue;
                }
                result.Add(new ScoredOption(options[s.Index], s.Score, s.Rationale));
            }
            return result;
        }
        catch (Exception ex)
        {
            _log?.Invoke($"[agent] {Name}: evaluate failed, declining: {ex.Message}");
            return Array.Empty<ScoredOption>();
        }
    }

    public void Dispose() => (_channel as IDisposable)?.Dispose();
}
