using System;
using System.Globalization;

namespace Godot;

[Serializable]
public struct Vector2I : IEquatable<Vector2I>
{
    public int X;
    public int Y;

    public enum Axis
    {
        X,
        Y
    }

    public Vector2I(int x, int y)
    {
        X = x;
        Y = y;
    }

    public int this[int index]
    {
        readonly get
        {
            return index switch
            {
                0 => X,
                1 => Y,
                _ => throw new ArgumentOutOfRangeException(nameof(index))
            };
        }
        set
        {
            switch (index)
            {
                case 0:
                    X = value;
                    break;
                case 1:
                    Y = value;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
    }

    private static readonly Vector2I _minValue = new Vector2I(int.MinValue, int.MinValue);
    private static readonly Vector2I _maxValue = new Vector2I(int.MaxValue, int.MaxValue);
    private static readonly Vector2I _zero = new Vector2I(0, 0);
    private static readonly Vector2I _one = new Vector2I(1, 1);
    private static readonly Vector2I _up = new Vector2I(0, -1);
    private static readonly Vector2I _down = new Vector2I(0, 1);
    private static readonly Vector2I _right = new Vector2I(1, 0);
    private static readonly Vector2I _left = new Vector2I(-1, 0);

    public static Vector2I MinValue => _minValue;
    public static Vector2I MaxValue => _maxValue;
    public static Vector2I Zero => _zero;
    public static Vector2I One => _one;
    public static Vector2I Up => _up;
    public static Vector2I Down => _down;
    public static Vector2I Right => _right;
    public static Vector2I Left => _left;

    public readonly void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;
    }

    public readonly Vector2I Abs() => new Vector2I(Mathf.Abs(X), Mathf.Abs(Y));

    public readonly float Aspect() => (float)X / Y;

    public readonly Vector2I Clamp(Vector2I min, Vector2I max)
    {
        return new Vector2I(Mathf.Clamp(X, min.X, max.X), Mathf.Clamp(Y, min.Y, max.Y));
    }

    public readonly Vector2I Clamp(int min, int max)
    {
        return new Vector2I(Mathf.Clamp(X, min, max), Mathf.Clamp(Y, min, max));
    }

    public readonly int DistanceSquaredTo(Vector2I to) => (to - this).LengthSquared();

    public readonly float DistanceTo(Vector2I to) => (to - this).Length();

    public readonly float Length()
    {
        int x = X;
        int y = Y;
        return Mathf.Sqrt(x * x + y * y);
    }

    public readonly int LengthSquared() => X * X + Y * Y;

    public readonly Vector2I Max(Vector2I with) => new Vector2I(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y));

    public readonly Vector2I Max(int with) => new Vector2I(Mathf.Max(X, with), Mathf.Max(Y, with));

    public readonly Vector2I Min(Vector2I with) => new Vector2I(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y));

    public readonly Vector2I Min(int with) => new Vector2I(Mathf.Min(X, with), Mathf.Min(Y, with));

    public readonly Axis MaxAxisIndex() => X >= Y ? Axis.X : Axis.Y;

    public readonly Axis MinAxisIndex() => X >= Y ? Axis.Y : Axis.X;

    public readonly Vector2I Sign() => new Vector2I(Mathf.Sign(X), Mathf.Sign(Y));

    public readonly Vector2I Snapped(Vector2I step)
    {
        return new Vector2I(
            (int)Mathf.Snapped((double)X, (double)step.X),
            (int)Mathf.Snapped((double)Y, (double)step.Y));
    }

    public readonly Vector2I Snapped(int step)
    {
        return new Vector2I(
            (int)Mathf.Snapped((double)X, (double)step),
            (int)Mathf.Snapped((double)Y, (double)step));
    }

    public readonly bool Equals(Vector2I other) => X == other.X && Y == other.Y;

    public override readonly bool Equals(object? obj) => obj is Vector2I other && Equals(other);

    public override readonly int GetHashCode() => HashCode.Combine(X, Y);

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)})";
    }

    public static Vector2I operator +(Vector2I left, Vector2I right)
    {
        return new Vector2I(left.X + right.X, left.Y + right.Y);
    }

    public static Vector2I operator -(Vector2I left, Vector2I right)
    {
        return new Vector2I(left.X - right.X, left.Y - right.Y);
    }

    public static Vector2I operator -(Vector2I vec) => new Vector2I(-vec.X, -vec.Y);

    public static Vector2I operator *(Vector2I vec, int scale)
    {
        return new Vector2I(vec.X * scale, vec.Y * scale);
    }

    public static Vector2I operator *(int scale, Vector2I vec)
    {
        return new Vector2I(vec.X * scale, vec.Y * scale);
    }

    public static Vector2I operator *(Vector2I left, Vector2I right)
    {
        return new Vector2I(left.X * right.X, left.Y * right.Y);
    }

    public static Vector2I operator /(Vector2I vec, int divisor)
    {
        return new Vector2I(vec.X / divisor, vec.Y / divisor);
    }

    public static Vector2I operator /(Vector2I vec, Vector2I divisorv)
    {
        return new Vector2I(vec.X / divisorv.X, vec.Y / divisorv.Y);
    }

    public static Vector2I operator %(Vector2I vec, int divisor)
    {
        return new Vector2I(vec.X % divisor, vec.Y % divisor);
    }

    public static Vector2I operator %(Vector2I vec, Vector2I divisorv)
    {
        return new Vector2I(vec.X % divisorv.X, vec.Y % divisorv.Y);
    }

    public static bool operator ==(Vector2I left, Vector2I right) => left.Equals(right);

    public static bool operator !=(Vector2I left, Vector2I right) => !left.Equals(right);

    public static bool operator <(Vector2I left, Vector2I right)
    {
        return left.X == right.X ? left.Y < right.Y : left.X < right.X;
    }

    public static bool operator >(Vector2I left, Vector2I right)
    {
        return left.X == right.X ? left.Y > right.Y : left.X > right.X;
    }

    public static bool operator <=(Vector2I left, Vector2I right)
    {
        return left.X == right.X ? left.Y <= right.Y : left.X < right.X;
    }

    public static bool operator >=(Vector2I left, Vector2I right)
    {
        return left.X == right.X ? left.Y >= right.Y : left.X > right.X;
    }

    public static implicit operator Vector2(Vector2I value)
    {
        return new Vector2(value.X, value.Y);
    }

    public static explicit operator Vector2I(Vector2 value)
    {
        return new Vector2I((int)value.X, (int)value.Y);
    }
}
