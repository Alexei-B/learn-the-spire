using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Logging;

namespace Lts2.Harness.Tests;

/// <summary>
/// Captures <see cref="LogLevel.Error"/> log entries the game emits while this sink is alive. The
/// game swallows exceptions on fire-and-forget tasks (combat/enemy-turn/event-option) via
/// <c>TaskHelper.RunSafely</c>, logging them through the static <c>Log.LogCallback</c> rather than
/// throwing — so a faulted effect (e.g. an NRE on a null UI singleton) leaves the run looking fine.
/// A test wraps a run in this sink and asserts <see cref="Errors"/> is empty to turn those otherwise
/// invisible faults into failures.
///
/// Callbacks fire on background threads, so collection is locked. The static event is process-wide;
/// since tests run sequentially, only one sink should be alive at a time (dispose it before the next).
/// </summary>
internal sealed class LogErrorSink : IDisposable
{
    private readonly object _gate = new();
    private readonly List<string> _errors = new();

    public LogErrorSink()
    {
        Log.LogCallback += OnLog;
    }

    private void OnLog(LogLevel level, string message, int skipFrames)
    {
        if (level < LogLevel.Error)
        {
            return;
        }
        lock (_gate)
        {
            _errors.Add(message);
        }
    }

    /// <summary>A snapshot of the error messages captured so far.</summary>
    public IReadOnlyList<string> Errors
    {
        get
        {
            lock (_gate)
            {
                return _errors.ToArray();
            }
        }
    }

    public void Dispose() => Log.LogCallback -= OnLog;
}
