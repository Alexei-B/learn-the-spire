using System;
using System.IO;

namespace Lts2.Agent.Wire;

/// <summary>
/// A minimal duplex "one JSON object per line" transport. The wire protocol is transport-agnostic:
/// today the only implementation is <see cref="StreamLineChannel"/> over a child process's stdio,
/// but the same framing works over a TCP socket, a named pipe, or an in-memory pipe (used by tests).
/// Both the training environment server and the evaluation decision-engine client read and write
/// through this seam so nothing else has to know how the bytes move.
/// </summary>
public interface ILineChannel
{
    /// <summary>Read the next line, or null at end of stream. Blocks until a line or EOF.</summary>
    string? ReadLine();

    /// <summary>Write one line (a single JSON message) and flush it immediately.</summary>
    void WriteLine(string line);
}

/// <summary>
/// An <see cref="ILineChannel"/> over a <see cref="TextReader"/>/<see cref="TextWriter"/> pair. Lines
/// are terminated with <c>"\n"</c> (not the platform default) so a Python <c>readline()</c> on the
/// other end sees the same framing on every OS, and each write is flushed so the peer never waits on
/// a buffered response.
/// </summary>
public sealed class StreamLineChannel : ILineChannel
{
    private readonly TextReader _reader;
    private readonly TextWriter _writer;

    public StreamLineChannel(TextReader reader, TextWriter writer)
    {
        _reader = reader ?? throw new ArgumentNullException(nameof(reader));
        _writer = writer ?? throw new ArgumentNullException(nameof(writer));
        _writer.NewLine = "\n";
    }

    public string? ReadLine() => _reader.ReadLine();

    public void WriteLine(string line)
    {
        _writer.WriteLine(line);
        _writer.Flush();
    }
}
