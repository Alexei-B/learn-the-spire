namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>ResourceLoader</c> facade. The harness ships no
/// packed resources (the 1.9 GB .pck is absent), so resource probes report "missing" and
/// loads return null. Game logic that asks whether an asset exists (e.g. picking an icon
/// path) degrades to its fallback. Grown on demand from real load errors; mirror the real
/// signatures so sts2 binds by name.
/// </summary>
public static class ResourceLoader
{
    public enum ThreadLoadStatus : long
    {
        InvalidResource,
        InProgress,
        Failed,
        Loaded,
    }

    public enum CacheMode : long
    {
        Ignore,
        Reuse,
        Replace,
        IgnoreDeep,
        ReplaceDeep,
    }

    /// <summary>No packed resources are present headless, so nothing exists.</summary>
    public static bool Exists(string path, string typeHint = "") => false;

    /// <summary>No packed resources are present headless, so loads return null.</summary>
    public static Resource? Load(string path, string typeHint = "", CacheMode cacheMode = CacheMode.Reuse) => null;

    /// <summary>No packed resources are present headless, so typed loads return null.</summary>
    public static T? Load<T>(string path, string? typeHint = null, CacheMode cacheMode = CacheMode.Reuse) where T : class => null;
}
