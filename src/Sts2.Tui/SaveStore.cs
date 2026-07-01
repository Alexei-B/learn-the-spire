using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Sts2.Harness;

namespace Sts2.Tui;

/// <summary>
/// Persists runs to disk as save files, so a run can be resumed later. Each file wraps the game's own
/// run save JSON (<see cref="GameHost.ToSaveJson"/>) plus a little display metadata (seed, character,
/// act/floor). Saving relies on the harness snapshot, which is only valid out of combat — so the app
/// autosaves when the player reaches the map (between rooms), and manual saves are offered there too.
/// </summary>
internal static class SaveStore
{
    private sealed record SaveWrapper(
        string Seed, string Character, int Act, int Floor, string Phase, string SavedUtc, string Run);

    /// <summary>Display metadata for a save file, without loading its (large) run.</summary>
    public sealed record SaveInfo(
        string Path, string Seed, string Character, int Act, int Floor, string Phase, DateTime SavedUtc, bool IsAutosave)
    {
        public string Describe() =>
            $"{Character}  ·  Act {Act} Floor {Floor}  ·  {Phase}  ·  seed {Seed}  ·  {SavedUtc.ToLocalTime():g}"
            + (IsAutosave ? "  (autosave)" : "");
    }

    private static readonly JsonSerializerOptions Opts = new() { WriteIndented = false };

    private static string Dir
    {
        get
        {
            string dir = System.IO.Path.Combine(AppContext.BaseDirectory, "saves");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string AutosavePath => System.IO.Path.Combine(Dir, "autosave.json");

    public static bool HasAutosave => File.Exists(AutosavePath);

    /// <summary>Write the run to the autosave slot. Silently a no-op if the state can't be snapshotted.</summary>
    public static void Autosave(GameHost host, GameState state)
    {
        try
        {
            Write(AutosavePath, host, state);
        }
        catch
        {
            // Best-effort: never let an autosave failure interrupt play.
        }
    }

    /// <summary>Write the run to a named manual slot. Throws on failure so the UI can report it.</summary>
    public static void Save(GameHost host, GameState state, string name)
    {
        string safe = new string(name.Where(c => char.IsLetterOrDigit(c) || c is '-' or '_' or ' ').ToArray()).Trim();
        if (string.IsNullOrEmpty(safe))
        {
            safe = "save";
        }
        Write(System.IO.Path.Combine(Dir, safe + ".json"), host, state);
    }

    private static void Write(string path, GameHost host, GameState state)
    {
        var wrapper = new SaveWrapper(
            state.Seed,
            state.Players.Count > 0 ? state.Players[0].Character : "?",
            state.ActIndex + 1,
            state.Floor,
            state.Phase.ToString(),
            DateTime.UtcNow.ToString("o"),
            host.ToSaveJson());
        File.WriteAllText(path, JsonSerializer.Serialize(wrapper, Opts));
    }

    /// <summary>All save files, newest first.</summary>
    public static IReadOnlyList<SaveInfo> List()
    {
        var infos = new List<SaveInfo>();
        foreach (string path in Directory.EnumerateFiles(Dir, "*.json"))
        {
            if (TryReadInfo(path) is { } info)
            {
                infos.Add(info);
            }
        }
        return infos.OrderByDescending(i => i.SavedUtc).ToList();
    }

    private static SaveInfo? TryReadInfo(string path)
    {
        try
        {
            SaveWrapper? w = JsonSerializer.Deserialize<SaveWrapper>(File.ReadAllText(path), Opts);
            if (w is null)
            {
                return null;
            }
            DateTime.TryParse(w.SavedUtc, null, System.Globalization.DateTimeStyles.RoundtripKind, out DateTime saved);
            return new SaveInfo(
                path, w.Seed, w.Character, w.Act, w.Floor, w.Phase, saved,
                string.Equals(System.IO.Path.GetFileName(path), "autosave.json", StringComparison.OrdinalIgnoreCase));
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Restore the run from a save file, replacing any run in progress. Throws on a bad file.</summary>
    public static GameHost Load(string path)
    {
        SaveWrapper w = JsonSerializer.Deserialize<SaveWrapper>(File.ReadAllText(path), Opts)
            ?? throw new InvalidOperationException("Save file is empty or corrupt.");
        return GameHost.RestoreFromJson(w.Run, w.Seed);
    }
}
