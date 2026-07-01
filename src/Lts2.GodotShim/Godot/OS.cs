using System;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>OS</c> singleton facade. We are never the
/// editor and pass no engine command-line args, so the members the game queries
/// return inert defaults. Grow on demand.
/// </summary>
public static class OS
{
    public static string[] GetCmdlineArgs() => Array.Empty<string>();

    public static string[] GetCmdlineUserArgs() => Array.Empty<string>();

    public static bool HasFeature(string tagName) => false;

    public static bool IsDebugBuild() => false;

    public static string GetLocale() => "en_US";

    public static string GetLocaleLanguage() => "en";

    public static string GetName() => "Headless";

    public static string GetExecutablePath() => Environment.ProcessPath ?? string.Empty;

    public static int GetProcessId() => Environment.ProcessId;
}
