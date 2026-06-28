namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>ProjectSettings</c> facade. Path
/// globalization maps Godot virtual paths to the shim's temp filesystem
/// (see <see cref="GodotPath"/>). Grow on demand.
/// </summary>
public static class ProjectSettings
{
    public static string GlobalizePath(string path) => GodotPath.Globalize(path);

    public static string LocalizePath(string path) => path;
}
