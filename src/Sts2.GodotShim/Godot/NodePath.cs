using System;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>NodePath</c> (a parsed scene-tree path).
/// Backed by a plain string; only used on visual code paths that are skipped
/// headless, so no parsing semantics are needed.
/// </summary>
public sealed class NodePath : IEquatable<NodePath>
{
    private readonly string _path;

    public NodePath(string path) => _path = path ?? string.Empty;

    public static implicit operator NodePath(string from) => new(from);

    public static implicit operator string(NodePath? from) => from?._path ?? string.Empty;

    public bool IsEmpty => _path.Length == 0;

    public bool Equals(NodePath? other) => other is not null && string.Equals(_path, other._path, StringComparison.Ordinal);

    public override bool Equals(object? obj) => obj is NodePath other && Equals(other);

    public override int GetHashCode() => _path.GetHashCode(StringComparison.Ordinal);

    public override string ToString() => _path;
}
