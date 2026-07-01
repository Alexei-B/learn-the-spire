using System;
using System.IO;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>GD</c> utility facade. Only the logging
/// entry points the game uses are provided; they route to the console instead of
/// the engine. Overload signatures mirror GodotSharp so sts2's compiled call sites
/// bind correctly.
/// </summary>
public static class GD
{
    // The game's logging (Log → ConsoleLogPrinter → GD.Print/PrintErr) is essentially the only
    // thing that writes to the terminal headless. By default it goes to the live console streams —
    // exactly the original behaviour, so tests still see the game's stdout/stderr. A host that owns
    // the terminal (e.g. a full-screen TUI) can redirect this chatter elsewhere (a log file) by
    // setting Out/Err, without globally redirecting Console (which a screen driver also uses).
    private static TextWriter? _out;
    private static TextWriter? _err;

    /// <summary>Where <c>Print</c>/<c>PrintRich</c> go. Defaults to the live <see cref="Console.Out"/>.</summary>
    public static TextWriter Out
    {
        get => _out ?? Console.Out;
        set => _out = value;
    }

    /// <summary>Where <c>PrintErr</c>/<c>PushError</c>/<c>PushWarning</c> go. Defaults to <see cref="Console.Error"/>.</summary>
    public static TextWriter Err
    {
        get => _err ?? Console.Error;
        set => _err = value;
    }

    public static void Print(string what) => Out.WriteLine(what);

    public static void Print(params object[] what) => Out.WriteLine(Join(what));

    public static void PrintRich(string what) => Out.WriteLine(what);

    public static void PrintRich(params object[] what) => Out.WriteLine(Join(what));

    public static void PrintErr(string what) => Err.WriteLine(what);

    public static void PrintErr(params object[] what) => Err.WriteLine(Join(what));

    public static void PushError(string message) => Err.WriteLine(message);

    public static void PushError(params object[] what) => Err.WriteLine(Join(what));

    public static void PushWarning(string message) => Err.WriteLine(message);

    public static void PushWarning(params object[] what) => Err.WriteLine(Join(what));

    private static string Join(object[] what) => what is null ? string.Empty : string.Concat(what);
}
