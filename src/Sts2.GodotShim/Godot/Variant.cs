using System;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>Variant</c>. The real type is a tagged union
/// marshalled to/from a native godot_variant; here it simply boxes a managed object.
/// This is sufficient because core game logic never round-trips values through
/// Variant — only infrastructure (signals, deferred calls, Godot collections) does,
/// and that only needs round-trip fidelity for plain managed values.
/// </summary>
public struct Variant : IDisposable
{
    public enum Type : long
    {
        Nil, Bool, Int, Float, String,
        Vector2, Vector2I, Rect2, Rect2I, Vector3, Vector3I, Transform2D,
        Vector4, Vector4I, Plane, Quaternion, Aabb, Basis, Transform3D, Projection,
        Color, StringName, NodePath, Rid, Object, Callable, Signal, Dictionary, Array,
        PackedByteArray, PackedInt32Array, PackedInt64Array, PackedFloat32Array,
        PackedFloat64Array, PackedStringArray, PackedVector2Array, PackedVector3Array,
        PackedColorArray, PackedVector4Array, Max,
    }

    private object? _obj;

    public readonly object? Obj => _obj;

    public static Variant From<T>(in T value) => new() { _obj = value };

    public static Variant CreateFrom<T>(in T value) => new() { _obj = value };

    // Implicit conversions from variant-compatible types. The real Variant has the
    // full set; we add them as the game's (mostly visual, dead-headless) call sites
    // require. Each just boxes the value.
    public static implicit operator Variant(bool from) => new() { _obj = from };
    public static implicit operator Variant(int from) => new() { _obj = from };
    public static implicit operator Variant(long from) => new() { _obj = from };
    public static implicit operator Variant(float from) => new() { _obj = from };
    public static implicit operator Variant(double from) => new() { _obj = from };
    public static implicit operator Variant(string from) => new() { _obj = from };
    public static implicit operator Variant(Vector2 from) => new() { _obj = from };
    public static implicit operator Variant(Vector2I from) => new() { _obj = from };
    public static implicit operator Variant(Color from) => new() { _obj = from };
    public static implicit operator Variant(StringName from) => new() { _obj = from };
    public static implicit operator Variant(NodePath from) => new() { _obj = from };
    public static implicit operator Variant(GodotObject from) => new() { _obj = from };

    public readonly T As<T>() => _obj is T t ? t : default!;

    public readonly object? AsSystemObject() => _obj;

    public readonly Type VariantType => _obj is null ? Type.Nil : Type.Object;

    public void Dispose() => _obj = null;

    public override readonly string ToString() => _obj?.ToString() ?? string.Empty;
}
