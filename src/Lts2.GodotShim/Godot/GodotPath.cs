using System;
using System.IO;

namespace Godot;

/// <summary>
/// Resolves Godot virtual paths (<c>user://</c>, <c>res://</c>) to real on-disk
/// paths for the headless shim. <c>user://</c> maps to a per-process temp directory
/// so saves/localization-overrides behave like the real game without touching the
/// user's profile. Override <see cref="UserDir"/> before first use to relocate.
/// </summary>
internal static class GodotPath
{
    public static string UserDir { get; set; } =
        Path.Combine(Path.GetTempPath(), "sts2-headless", "user");

    public static string ResDir { get; set; } =
        Path.Combine(AppContext.BaseDirectory, "res");

    public static string Globalize(string path)
    {
        if (path.StartsWith("user://", StringComparison.Ordinal))
        {
            return Path.Combine(UserDir, path.Substring("user://".Length).Replace('/', Path.DirectorySeparatorChar));
        }
        if (path.StartsWith("res://", StringComparison.Ordinal))
        {
            return Path.Combine(ResDir, path.Substring("res://".Length).Replace('/', Path.DirectorySeparatorChar));
        }
        return path;
    }
}
