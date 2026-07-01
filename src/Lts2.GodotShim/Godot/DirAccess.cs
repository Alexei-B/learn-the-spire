using System;
using System.IO;
using System.Linq;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>DirAccess</c>, backed by System.IO over the
/// shim's globalized temp filesystem (see <see cref="GodotPath"/>). Exposes both the
/// static helpers and the instance directory-listing API the game uses.
///
/// <see cref="Open"/> returns a (possibly empty) instance rather than null for a
/// missing directory — headless we have no res:// localization tree, and callers
/// treat an empty listing as "no files" which is the correct mechanical behavior.
/// </summary>
public sealed class DirAccess : IDisposable
{
    private readonly string[] _files;
    private readonly string[] _dirs;

    private DirAccess(string realDir)
    {
        if (Directory.Exists(realDir))
        {
            _files = Directory.GetFiles(realDir).Select(Path.GetFileName).Where(n => n != null).ToArray()!;
            _dirs = Directory.GetDirectories(realDir).Select(Path.GetFileName).Where(n => n != null).ToArray()!;
        }
        else
        {
            _files = Array.Empty<string>();
            _dirs = Array.Empty<string>();
        }
    }

    public static DirAccess Open(string path) => new(GodotPath.Globalize(path));

    public string[] GetFiles() => _files;

    public string[] GetDirectories() => _dirs;

    public void Dispose() { }

    // ---- static helpers ----

    public static bool DirExistsAbsolute(string path) => Directory.Exists(GodotPath.Globalize(path));

    public static Error MakeDirAbsolute(string path) => MakeDirRecursiveAbsolute(path);

    public static Error MakeDirRecursiveAbsolute(string path)
    {
        Directory.CreateDirectory(GodotPath.Globalize(path));
        return Error.Ok;
    }

    public static bool FileExists(string path) => File.Exists(GodotPath.Globalize(path));
}
