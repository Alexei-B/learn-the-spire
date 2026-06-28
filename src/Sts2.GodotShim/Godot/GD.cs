using System;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>GD</c> utility facade. Only the logging
/// entry points the game uses are provided; they route to the console instead of
/// the engine. Overload signatures mirror GodotSharp so sts2's compiled call sites
/// bind correctly.
/// </summary>
public static class GD
{
    public static void Print(string what) => Console.Out.WriteLine(what);

    public static void Print(params object[] what) => Console.Out.WriteLine(Join(what));

    public static void PrintRich(string what) => Console.Out.WriteLine(what);

    public static void PrintRich(params object[] what) => Console.Out.WriteLine(Join(what));

    public static void PrintErr(string what) => Console.Error.WriteLine(what);

    public static void PrintErr(params object[] what) => Console.Error.WriteLine(Join(what));

    public static void PushError(string message) => Console.Error.WriteLine(message);

    public static void PushError(params object[] what) => Console.Error.WriteLine(Join(what));

    public static void PushWarning(string message) => Console.Error.WriteLine(message);

    public static void PushWarning(params object[] what) => Console.Error.WriteLine(Join(what));

    private static string Join(object[] what) => what is null ? string.Empty : string.Concat(what);
}
