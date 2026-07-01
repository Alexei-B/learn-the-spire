using System;
using System.IO;
using System.Text;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>FileAccess</c>, backed by System.IO over the
/// shim's globalized temp filesystem (see <see cref="GodotPath"/>). Only the members
/// the game's save layer uses are implemented; grow on demand.
///
/// Mirrors Godot's quirk that <see cref="Open"/> returns null on failure and stashes
/// the reason in a thread-static slot read back via <see cref="GetOpenError"/>.
/// </summary>
public sealed class FileAccess : IDisposable
{
    public enum ModeFlags : long
    {
        Read = 1L,
        Write = 2L,
        ReadWrite = 3L,
        WriteRead = 7L,
    }

    [ThreadStatic] private static Error _lastOpenError;

    private readonly FileStream _stream;
    private Error _error = Error.Ok;

    private FileAccess(FileStream stream) => _stream = stream;

    public static FileAccess? Open(string path, ModeFlags flags)
    {
        string real = GodotPath.Globalize(path);
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(real)!);
            FileStream stream = flags switch
            {
                ModeFlags.Read => new FileStream(real, FileMode.Open, FileAccessMode(flags), FileShare.ReadWrite),
                ModeFlags.Write => new FileStream(real, FileMode.Create, FileAccessMode(flags), FileShare.ReadWrite),
                _ => new FileStream(real, FileMode.OpenOrCreate, FileAccessMode(flags), FileShare.ReadWrite),
            };
            _lastOpenError = Error.Ok;
            return new FileAccess(stream);
        }
        catch (FileNotFoundException)
        {
            _lastOpenError = Error.FileNotFound;
            return null;
        }
        catch (DirectoryNotFoundException)
        {
            _lastOpenError = Error.FileNotFound;
            return null;
        }
        catch (Exception)
        {
            _lastOpenError = Error.Failed;
            return null;
        }
    }

    private static System.IO.FileAccess FileAccessMode(ModeFlags flags) => flags switch
    {
        ModeFlags.Read => System.IO.FileAccess.Read,
        ModeFlags.Write => System.IO.FileAccess.Write,
        _ => System.IO.FileAccess.ReadWrite,
    };

    public static Error GetOpenError() => _lastOpenError;

    public static bool FileExists(string path) => File.Exists(GodotPath.Globalize(path));

    public static ulong GetModifiedTime(string file)
    {
        string real = GodotPath.Globalize(file);
        return File.Exists(real)
            ? (ulong)((DateTimeOffset)File.GetLastWriteTimeUtc(real)).ToUnixTimeSeconds()
            : 0UL;
    }

    public static long GetSize(string file)
    {
        string real = GodotPath.Globalize(file);
        return File.Exists(real) ? new FileInfo(real).Length : 0L;
    }

    public Error GetError() => _error;

    public ulong GetLength() => (ulong)_stream.Length;

    public ulong GetPosition() => (ulong)_stream.Position;

    public void Seek(ulong position) => _stream.Seek((long)position, SeekOrigin.Begin);

    public byte[] GetBuffer(long length)
    {
        long clamped = Math.Max(0, Math.Min(length, _stream.Length - _stream.Position));
        var buffer = new byte[clamped];
        int read = _stream.Read(buffer, 0, (int)clamped);
        if (read == buffer.Length)
        {
            return buffer;
        }
        Array.Resize(ref buffer, read);
        return buffer;
    }

    public string GetAsText(bool skipCr = false)
    {
        long pos = _stream.Position;
        _stream.Seek(0, SeekOrigin.Begin);
        using var reader = new StreamReader(_stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
        string text = reader.ReadToEnd();
        _stream.Seek(pos, SeekOrigin.Begin);
        return skipCr ? text.Replace("\r", string.Empty) : text;
    }

    public bool StoreBuffer(byte[] buffer)
    {
        _stream.Write(buffer, 0, buffer.Length);
        return true;
    }

    public bool StoreBuffer(ReadOnlySpan<byte> buffer)
    {
        _stream.Write(buffer);
        return true;
    }

    public bool StoreString(string @string)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(@string);
        _stream.Write(bytes, 0, bytes.Length);
        return true;
    }

    public void Flush() => _stream.Flush();

    public void Close() => _stream.Dispose();

    public void Dispose() => _stream.Dispose();
}
