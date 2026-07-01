using System;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>StringName</c> (an interned string handle in
/// the engine). Backed by a plain managed string with value equality, which is all
/// the game relies on for its cached method/property/signal name lookups.
/// </summary>
public sealed class StringName : IEquatable<StringName>
{
    private readonly string _value;

    public StringName(string value) => _value = value ?? string.Empty;

    public static implicit operator StringName(string from) => new(from);

    public static implicit operator string(StringName from) => from?._value ?? string.Empty;

    public bool IsEmpty => _value.Length == 0;

    public bool Equals(StringName? other) => other is not null && string.Equals(_value, other._value, StringComparison.Ordinal);

    public override bool Equals(object? obj) => obj is StringName other && Equals(other);

    public static bool operator ==(StringName? left, StringName? right) =>
        left is null ? right is null : left.Equals(right);

    public static bool operator !=(StringName? left, StringName? right) => !(left == right);

    public override int GetHashCode() => _value.GetHashCode(StringComparison.Ordinal);

    public override string ToString() => _value;
}
