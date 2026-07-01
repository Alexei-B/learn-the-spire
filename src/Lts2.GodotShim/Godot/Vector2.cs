using System;
using System.Globalization;

namespace Godot;

[Serializable]
public struct Vector2 : IEquatable<Vector2>
{
    public float X;
    public float Y;

    public enum Axis
    {
        X,
        Y
    }

    public Vector2(float x, float y)
    {
        X = x;
        Y = y;
    }

    public float this[int index]
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

    private static readonly Vector2 _zero = new Vector2(0f, 0f);
    private static readonly Vector2 _one = new Vector2(1f, 1f);
    private static readonly Vector2 _inf = new Vector2(float.PositiveInfinity, float.PositiveInfinity);
    private static readonly Vector2 _up = new Vector2(0f, -1f);
    private static readonly Vector2 _down = new Vector2(0f, 1f);
    private static readonly Vector2 _right = new Vector2(1f, 0f);
    private static readonly Vector2 _left = new Vector2(-1f, 0f);

    public static Vector2 Zero => _zero;
    public static Vector2 One => _one;
    public static Vector2 Inf => _inf;
    public static Vector2 Up => _up;
    public static Vector2 Down => _down;
    public static Vector2 Right => _right;
    public static Vector2 Left => _left;

    public static Vector2 FromAngle(float angle)
    {
        return new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
    }

    public readonly void Deconstruct(out float x, out float y)
    {
        x = X;
        y = Y;
    }

    internal void Normalize()
    {
        float lengthSquared = LengthSquared();
        if (lengthSquared == 0f)
        {
            X = 0f;
            Y = 0f;
        }
        else
        {
            float length = Mathf.Sqrt(lengthSquared);
            X /= length;
            Y /= length;
        }
    }

    public readonly Vector2 Abs() => new Vector2(Mathf.Abs(X), Mathf.Abs(Y));

    public readonly Vector2 Ceil() => new Vector2(Mathf.Ceil(X), Mathf.Ceil(Y));

    public readonly Vector2 Floor() => new Vector2(Mathf.Floor(X), Mathf.Floor(Y));

    public readonly Vector2 Round() => new Vector2(Mathf.Round(X), Mathf.Round(Y));

    public readonly Vector2 Sign() => new Vector2(Mathf.Sign(X), Mathf.Sign(Y));

    public readonly float Angle() => Mathf.Atan2(Y, X);

    public readonly float AngleTo(Vector2 to) => Mathf.Atan2(Cross(to), Dot(to));

    public readonly float AngleToPoint(Vector2 to) => Mathf.Atan2(to.Y - Y, to.X - X);

    public readonly float Aspect() => X / Y;

    public readonly Vector2 Bounce(Vector2 normal) => -Reflect(normal);

    public readonly Vector2 Clamp(Vector2 min, Vector2 max)
    {
        return new Vector2(Mathf.Clamp(X, min.X, max.X), Mathf.Clamp(Y, min.Y, max.Y));
    }

    public readonly Vector2 Clamp(float min, float max)
    {
        return new Vector2(Mathf.Clamp(X, min, max), Mathf.Clamp(Y, min, max));
    }

    public readonly float Cross(Vector2 with) => X * with.Y - Y * with.X;

    public readonly Vector2 CubicInterpolate(Vector2 b, Vector2 preA, Vector2 postB, float weight)
    {
        return new Vector2(
            Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight),
            Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight));
    }

    public readonly Vector2 CubicInterpolateInTime(Vector2 b, Vector2 preA, Vector2 postB, float weight, float t, float preAT, float postBT)
    {
        return new Vector2(
            Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT),
            Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT));
    }

    public readonly Vector2 BezierInterpolate(Vector2 control1, Vector2 control2, Vector2 end, float t)
    {
        return new Vector2(
            Mathf.BezierInterpolate(X, control1.X, control2.X, end.X, t),
            Mathf.BezierInterpolate(Y, control1.Y, control2.Y, end.Y, t));
    }

    public readonly Vector2 BezierDerivative(Vector2 control1, Vector2 control2, Vector2 end, float t)
    {
        return new Vector2(
            Mathf.BezierDerivative(X, control1.X, control2.X, end.X, t),
            Mathf.BezierDerivative(Y, control1.Y, control2.Y, end.Y, t));
    }

    public readonly Vector2 DirectionTo(Vector2 to) => (to - this).Normalized();

    public readonly float DistanceSquaredTo(Vector2 to)
    {
        return (X - to.X) * (X - to.X) + (Y - to.Y) * (Y - to.Y);
    }

    public readonly float DistanceTo(Vector2 to) => Mathf.Sqrt(DistanceSquaredTo(to));

    public readonly float Dot(Vector2 with) => X * with.X + Y * with.Y;

    public readonly Vector2 Inverse() => new Vector2(1f / X, 1f / Y);

    public readonly bool IsFinite() => Mathf.IsFinite(X) && Mathf.IsFinite(Y);

    public readonly bool IsNormalized() => Mathf.Abs(LengthSquared() - 1f) < 1e-6f;

    public readonly float Length() => Mathf.Sqrt(X * X + Y * Y);

    public readonly float LengthSquared() => X * X + Y * Y;

    public readonly Vector2 Lerp(Vector2 to, float weight)
    {
        return new Vector2(Mathf.Lerp(X, to.X, weight), Mathf.Lerp(Y, to.Y, weight));
    }

    public readonly Vector2 LimitLength(float length = 1f)
    {
        float currentLength = Length();
        Vector2 result = this;
        if (currentLength > 0f && length < currentLength)
        {
            result /= currentLength;
            result *= length;
        }

        return result;
    }

    public readonly Vector2 Max(Vector2 with) => new Vector2(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y));

    public readonly Vector2 Max(float with) => new Vector2(Mathf.Max(X, with), Mathf.Max(Y, with));

    public readonly Vector2 Min(Vector2 with) => new Vector2(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y));

    public readonly Vector2 Min(float with) => new Vector2(Mathf.Min(X, with), Mathf.Min(Y, with));

    public readonly Axis MaxAxisIndex() => X < Y ? Axis.Y : Axis.X;

    public readonly Axis MinAxisIndex() => X < Y ? Axis.X : Axis.Y;

    public readonly Vector2 MoveToward(Vector2 to, float delta)
    {
        Vector2 offset = to - this;
        float distance = offset.Length();
        if (distance <= delta || distance < 1e-6f)
        {
            return to;
        }

        return this + offset / distance * delta;
    }

    public readonly Vector2 Normalized()
    {
        Vector2 result = this;
        result.Normalize();
        return result;
    }

    public readonly Vector2 PosMod(float mod)
    {
        return new Vector2(Mathf.PosMod(X, mod), Mathf.PosMod(Y, mod));
    }

    public readonly Vector2 PosMod(Vector2 modv)
    {
        return new Vector2(Mathf.PosMod(X, modv.X), Mathf.PosMod(Y, modv.Y));
    }

    public readonly Vector2 Project(Vector2 onNormal)
    {
        return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
    }

    public readonly Vector2 Reflect(Vector2 normal)
    {
        return 2f * Dot(normal) * normal - this;
    }

    public readonly Vector2 Rotated(float angle)
    {
        (float sin, float cos) = Mathf.SinCos(angle);
        return new Vector2(X * cos - Y * sin, X * sin + Y * cos);
    }

    public readonly Vector2 Slerp(Vector2 to, float weight)
    {
        float startLengthSquared = LengthSquared();
        float endLengthSquared = to.LengthSquared();
        if (startLengthSquared == 0f || endLengthSquared == 0f)
        {
            return Lerp(to, weight);
        }

        float startLength = Mathf.Sqrt(startLengthSquared);
        float resultLength = Mathf.Lerp(startLength, Mathf.Sqrt(endLengthSquared), weight);
        float angle = AngleTo(to);
        return Rotated(angle * weight) * (resultLength / startLength);
    }

    public readonly Vector2 Slide(Vector2 normal) => this - normal * Dot(normal);

    public readonly Vector2 Snapped(Vector2 step)
    {
        return new Vector2(Mathf.Snapped(X, step.X), Mathf.Snapped(Y, step.Y));
    }

    public readonly Vector2 Snapped(float step)
    {
        return new Vector2(Mathf.Snapped(X, step), Mathf.Snapped(Y, step));
    }

    public readonly Vector2 Orthogonal() => new Vector2(Y, -X);

    public readonly bool IsEqualApprox(Vector2 other)
    {
        return Mathf.IsEqualApprox(X, other.X) && Mathf.IsEqualApprox(Y, other.Y);
    }

    public readonly bool IsZeroApprox() => Mathf.IsZeroApprox(X) && Mathf.IsZeroApprox(Y);

    public readonly bool Equals(Vector2 other) => X == other.X && Y == other.Y;

    public override readonly bool Equals(object? obj) => obj is Vector2 other && Equals(other);

    public override readonly int GetHashCode() => HashCode.Combine(X, Y);

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)})";
    }

    public static Vector2 operator +(Vector2 left, Vector2 right)
    {
        return new Vector2(left.X + right.X, left.Y + right.Y);
    }

    public static Vector2 operator -(Vector2 left, Vector2 right)
    {
        return new Vector2(left.X - right.X, left.Y - right.Y);
    }

    public static Vector2 operator -(Vector2 vec) => new Vector2(-vec.X, -vec.Y);

    public static Vector2 operator *(Vector2 vec, float scale)
    {
        return new Vector2(vec.X * scale, vec.Y * scale);
    }

    public static Vector2 operator *(float scale, Vector2 vec)
    {
        return new Vector2(vec.X * scale, vec.Y * scale);
    }

    public static Vector2 operator *(Vector2 left, Vector2 right)
    {
        return new Vector2(left.X * right.X, left.Y * right.Y);
    }

    public static Vector2 operator /(Vector2 vec, float divisor)
    {
        return new Vector2(vec.X / divisor, vec.Y / divisor);
    }

    public static Vector2 operator /(Vector2 vec, Vector2 divisorv)
    {
        return new Vector2(vec.X / divisorv.X, vec.Y / divisorv.Y);
    }

    public static Vector2 operator %(Vector2 vec, float divisor)
    {
        return new Vector2(vec.X % divisor, vec.Y % divisor);
    }

    public static Vector2 operator %(Vector2 vec, Vector2 divisorv)
    {
        return new Vector2(vec.X % divisorv.X, vec.Y % divisorv.Y);
    }

    public static bool operator ==(Vector2 left, Vector2 right) => left.Equals(right);

    public static bool operator !=(Vector2 left, Vector2 right) => !left.Equals(right);

    public static bool operator <(Vector2 left, Vector2 right)
    {
        return left.X == right.X ? left.Y < right.Y : left.X < right.X;
    }

    public static bool operator >(Vector2 left, Vector2 right)
    {
        return left.X == right.X ? left.Y > right.Y : left.X > right.X;
    }

    public static bool operator <=(Vector2 left, Vector2 right)
    {
        return left.X == right.X ? left.Y <= right.Y : left.X < right.X;
    }

    public static bool operator >=(Vector2 left, Vector2 right)
    {
        return left.X == right.X ? left.Y >= right.Y : left.X > right.X;
    }
}
