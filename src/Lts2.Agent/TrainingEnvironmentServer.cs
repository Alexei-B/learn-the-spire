using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Lts2.Agent.Wire;
using Lts2.Harness;
using MegaCrit.Sts2.Core.Models;

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
            return (Error(ex.Message), false);
        }
    }

    private string Reset(EnvCommand command)
    {
        string seed = string.IsNullOrEmpty(command.Seed) ? "AGENT" : command.Seed!;
        int ascension = command.Ascension ?? 0;
        CharacterModel character = ResolveCharacter(command.Character);

        _host = GameHost.StartNewRun(seed, new[] { character }, ascension);
        _host.EnterFirstRoom();
        return Observe();
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
            IReadOnlyList<GameOption> options = host.ListOptions();
            if (index < 0 || index >= options.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(command.Index), index, $"Action index out of range (expected 0..{options.Count - 1}).");
            }
            host.Apply(options[index]);
        }
        else
        {
            throw new InvalidOperationException("A 'step' needs an 'index' or 'cardIndices'.");
        }

        return Observe();
    }

    private string Observe()
    {
        GameHost host = _host!;
        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();
        return JsonSerializer.Serialize(Observation.From(state, options), AgentJson.Options);
    }

    private static CharacterModel ResolveCharacter(string? name) =>
        string.IsNullOrEmpty(name)
            ? ModelDb.AllCharacters.First()
            : ModelDb.AllCharacters.First(
                c => c.Id.Entry.Contains(name!, StringComparison.OrdinalIgnoreCase));

    private static string Error(string message) =>
        JsonSerializer.Serialize(new ErrorResponse { Error = message }, AgentJson.Options);
}
