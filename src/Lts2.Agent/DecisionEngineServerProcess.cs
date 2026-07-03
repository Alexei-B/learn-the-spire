using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Lts2.Agent.Wire;

namespace Lts2.Agent;

/// <summary>
/// A synchronous request/response channel to a decision engine: send one request line, get one
/// response line (or null if the engine is unavailable). <see cref="DecisionEngineServerProcess"/> is
/// the production implementation (a child process over stdio); tests and alternative transports (TCP,
/// in-process) supply their own. <see cref="ProcessDecisionEngine"/> depends only on this seam.
/// </summary>
public interface IDecisionChannel
{
    /// <summary>Send one request line and return the response line, or null if unavailable.</summary>
    string? SendRequest(string requestLine);
}

/// <summary>
/// A generic manager for an external decision-engine process spoken to over its stdio with the
/// line protocol (<see cref="ILineChannel"/>). It launches the child, forwards its stderr to a log,
/// and exposes a single synchronous <see cref="SendRequest"/> (one request/response at a time, under
/// a lock, with a timeout). It knows nothing about game types — <see cref="ProcessDecisionEngine"/>
/// layers the <see cref="IDecisionEngine"/> contract on top. This is the reusable seam for plugging
/// in a decision engine trained in another language (e.g. a Python policy server).
///
/// <para>A misbehaving peer (a response that never comes) faults the manager: the process is killed
/// and every subsequent <see cref="SendRequest"/> returns null so the caller degrades gracefully
/// rather than hanging the UI.</para>
/// </summary>
public sealed class DecisionEngineServerProcess : IDecisionChannel, IDisposable
{
    private readonly Process _process;
    private readonly StreamLineChannel _channel;
    private readonly Action<string>? _log;
    private readonly TimeSpan _timeout;
    private readonly object _lock = new();
    private bool _faulted;
    private bool _disposed;

    private DecisionEngineServerProcess(Process process, Action<string>? log, TimeSpan timeout)
    {
        _process = process;
        _log = log;
        _timeout = timeout;
        _channel = new StreamLineChannel(process.StandardOutput, process.StandardInput);
    }

    /// <summary>
    /// Launch <paramref name="command"/> (with <paramref name="arguments"/>) as a child process,
    /// redirecting its stdio, and begin forwarding its stderr to <paramref name="log"/>. The child is
    /// expected to speak the line protocol on its stdout/stdin.
    /// </summary>
    /// <param name="timeout">Max wait for a single response before the peer is deemed broken; null =
    /// a default of 30s.</param>
    public static DecisionEngineServerProcess Start(
        string command,
        string? arguments = null,
        string? workingDirectory = null,
        TimeSpan? timeout = null,
        Action<string>? log = null)
    {
        if (string.IsNullOrWhiteSpace(command))
        {
            throw new ArgumentException("A command is required.", nameof(command));
        }

        var psi = new ProcessStartInfo
        {
            FileName = command,
            Arguments = arguments ?? string.Empty,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        if (!string.IsNullOrEmpty(workingDirectory))
        {
            psi.WorkingDirectory = workingDirectory;
        }

        Process process = Process.Start(psi)
            ?? throw new InvalidOperationException($"Failed to start decision-engine process '{command}'.");

        var manager = new DecisionEngineServerProcess(process, log, timeout ?? TimeSpan.FromSeconds(30));
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                log?.Invoke($"[agent stderr] {e.Data}");
            }
        };
        process.BeginErrorReadLine();
        return manager;
    }

    /// <summary>
    /// Send one request line and return the response line, or null if the peer is faulted, has exited,
    /// times out, or the manager is disposed. Serialized: only one request is in flight at a time.
    /// </summary>
    public string? SendRequest(string requestLine)
    {
        lock (_lock)
        {
            if (_disposed || _faulted)
            {
                return null;
            }
            if (_process.HasExited)
            {
                Fault("process has exited");
                return null;
            }

            try
            {
                _channel.WriteLine(requestLine);
                return ReadResponse();
            }
            catch (Exception ex)
            {
                Fault($"I/O error: {ex.Message}");
                return null;
            }
        }
    }

    private string? ReadResponse()
    {
        // StreamReader.ReadLine blocks and can't be cancelled, so read on a worker and bound the wait.
        // A peer that blows the timeout is treated as broken (killed) to avoid stream desync on the
        // next request.
        Task<string?> read = Task.Run(_channel.ReadLine);
        if (!read.Wait(_timeout))
        {
            Fault($"no response within {_timeout.TotalSeconds:0.#}s");
            return null;
        }
        string? line = read.Result;
        if (line is null)
        {
            Fault("peer closed its output stream");
        }
        return line;
    }

    private void Fault(string reason)
    {
        if (_faulted)
        {
            return;
        }
        _faulted = true;
        _log?.Invoke($"[agent] decision engine faulted: {reason}");
        TryKill();
    }

    private void TryKill()
    {
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (Exception ex)
        {
            _log?.Invoke($"[agent] failed to kill decision engine: {ex.Message}");
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
        }
        TryKill();
        _process.Dispose();
    }
}
