using System;
using System.Runtime.CompilerServices;

namespace Godot;

/// <summary>
/// Inert replacement for Godot's <c>SignalAwaiter</c>. The game only awaits engine
/// signals (e.g. SceneTree.ProcessFrame) on its non-headless code paths, which are
/// gated out by NonInteractiveMode. This exists so those branches JIT; it must never
/// actually be awaited headless.
/// </summary>
public sealed class SignalAwaiter : IAwaiter<Variant[]>, INotifyCompletion, IAwaitable<Variant[]>
{
    public bool IsCompleted => true;

    public void OnCompleted(Action continuation) =>
        throw new NotSupportedException("SignalAwaiter is not available in the headless shim (no frame loop).");

    public Variant[] GetResult() =>
        throw new NotSupportedException("SignalAwaiter is not available in the headless shim (no frame loop).");

    public IAwaiter<Variant[]> GetAwaiter() => this;
}

/// <summary>
/// Inert replacement for Godot's <c>Engine</c> singleton facade. Headless there is no
/// main loop; the only references to it are on NonInteractiveMode-gated branches.
/// </summary>
public static class Engine
{
    // Some live code paths (e.g. the reshuffle pacing loop) read the main loop's root
    // for frame timing, so return the shared scene tree rather than throwing.
    public static MainLoop GetMainLoop() => SceneTree.Shared;

    public static bool IsEditorHint() => false;

    public static ulong GetProcessFrames() => 0;
}
