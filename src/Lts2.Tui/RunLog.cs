using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Lts2.Harness;

namespace Lts2.Tui;

/// <summary>
/// A persistent, per-run log written to disk (under <c>logs/</c>), so a run that crashes, hangs, or
/// produces a corrupt save leaves a record that can be analysed afterwards. Unlike <see cref="GameLog"/>
/// — an in-memory scrollback that's cleared whenever a run is loaded — this survives the process.
///
/// It captures, in order: a run banner; every decision applied, <em>written and flushed before the
/// action runs</em> (so a hang on end-turn still records which action was in flight); the resulting
/// state and diffed events; any exception with its full stack; and — via <see cref="Writer"/> teed
/// into the game's GD output (see <see cref="TeeTextWriter"/>) — the engine's own log/error stream,
/// where a swallowed enemy-turn exception behind a hang tends to surface. Every write is flushed
/// immediately and guarded so logging can never interrupt play.
/// </summary>
internal sealed class RunLog : IDisposable
{
    private readonly StreamWriter _file;
    private readonly TextWriter _sync; // thread-safe: written from the UI thread and game threads

    private RunLog(StreamWriter file)
    {
        _file = file;
        _sync = TextWriter.Synchronized(file);
    }

    /// <summary>The run-log directory (created on demand).</summary>
    public static string Dir
    {
        get
        {
            string dir = Path.Combine(AppContext.BaseDirectory, "logs");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    /// <summary>The path of this run's log file.</summary>
    public string FilePath { get; private set; } = "";

    /// <summary>A synchronized writer suitable for teeing the game's GD.Print/GD.PrintErr output.</summary>
    public TextWriter Writer => _sync;

    /// <summary>Open a fresh run-log file. Returns null on failure — logging must never break play.</summary>
    public static RunLog? Create(string seed, string character)
    {
        try
        {
            string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            string name = $"run-{stamp}-{Safe(character)}-{Safe(seed)}.log";
            string path = Path.Combine(Dir, name);
            var sw = new StreamWriter(path, append: false) { AutoFlush = true };
            return new RunLog(sw) { FilePath = path };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>A run-level banner (run start / load).</summary>
    public void Banner(string line) => WriteLine($"==== {Stamp()}  {Flatten(line)} ====");

    /// <summary>A free-form note.</summary>
    public void Note(string note) => WriteLine($"{Stamp()}  ·  {Flatten(note)}");

    /// <summary>
    /// The decision about to be applied. Flushed <em>before</em> the action runs, so if the action
    /// hangs (e.g. an enemy turn that never ends) this line is the last one in the file, naming the
    /// action that got stuck.
    /// </summary>
    public void Action(string header) => WriteLine($"{Stamp()}  ▶  {Flatten(header)}");

    /// <summary>The state reached after an action, plus the human-readable events it produced.</summary>
    public void Result(GameState after, IReadOnlyList<Line> events)
    {
        var sb = new StringBuilder();
        sb.Append($"{Stamp()}     = {after.Phase} · A{after.ActIndex + 1} F{after.Floor}");
        if (after.Players.Count > 0)
        {
            PlayerState p = after.Players[0];
            sb.Append($" · HP {p.CurrentHp}/{p.MaxHp} · {p.Gold}g · score {after.Score}");
        }
        WriteLine(sb.ToString());
        foreach (Line l in events)
        {
            string text = Flatten(l);
            if (text.Length > 0)
            {
                WriteLine("          " + text);
            }
        }
    }

    /// <summary>An exception thrown while applying an action — logged with its full stack.</summary>
    public void Error(Exception ex)
    {
        WriteLine($"{Stamp()}  !! ERROR  {ex.GetType().Name}: {ex.Message}");
        WriteLine(ex.ToString());
    }

    public void Dispose()
    {
        try
        {
            _file.Flush();
            _file.Dispose();
        }
        catch
        {
            // Best-effort close.
        }
    }

    private void WriteLine(string s)
    {
        try
        {
            _sync.WriteLine(s);
        }
        catch
        {
            // Never let logging throw into gameplay.
        }
    }

    private static string Stamp() => DateTime.Now.ToString("HH:mm:ss.fff");

    private static string Flatten(Line line) => string.Concat(line.Select(s => s.Text)).TrimEnd();

    private static string Flatten(string s) => s.Replace("\r", " ").Replace("\n", " ").Trim();

    private static string Safe(string s)
    {
        var chars = s.Where(c => char.IsLetterOrDigit(c) || c is '-' or '_').ToArray();
        string cleaned = new string(chars);
        return string.IsNullOrEmpty(cleaned) ? "x" : cleaned;
    }
}

/// <summary>
/// A <see cref="TextWriter"/> that fans every write out to two targets, swallowing per-target
/// failures. Used to route the game's GD.Print/GD.PrintErr stream to both the shared session log and
/// the current run's <see cref="RunLog"/> at once. The game only ever calls <c>WriteLine(string)</c>
/// on these writers, but the core write methods are overridden too for safety.
/// </summary>
internal sealed class TeeTextWriter : TextWriter
{
    private readonly TextWriter _a;
    private readonly TextWriter _b;

    public TeeTextWriter(TextWriter a, TextWriter b)
    {
        _a = a;
        _b = b;
    }

    public override Encoding Encoding => _a.Encoding;

    public override void WriteLine(string? value)
    {
        Safe(() => _a.WriteLine(value));
        Safe(() => _b.WriteLine(value));
    }

    public override void Write(string? value)
    {
        Safe(() => _a.Write(value));
        Safe(() => _b.Write(value));
    }

    public override void Write(char value)
    {
        Safe(() => _a.Write(value));
        Safe(() => _b.Write(value));
    }

    public override void Flush()
    {
        Safe(() => _a.Flush());
        Safe(() => _b.Flush());
    }

    private static void Safe(Action write)
    {
        try
        {
            write();
        }
        catch
        {
            // A dead target (e.g. a run log disposed at run switch) must not break the other.
        }
    }
}
