using System.Diagnostics;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>Time</c> singleton facade. Backed by a
/// process Stopwatch; only the millisecond tick counter the game uses is provided.
/// </summary>
public static class Time
{
    private static readonly Stopwatch Clock = Stopwatch.StartNew();

    public static ulong GetTicksMsec() => (ulong)Clock.ElapsedMilliseconds;

    public static ulong GetTicksUsec() => (ulong)(Clock.Elapsed.Ticks / 10);
}
