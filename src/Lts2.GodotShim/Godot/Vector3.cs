using System;
using System.Globalization;

namespace Godot;

[Serializable]
public struct Vector3 : IEquatable<Vector3>
{
    public float X;
    public float Y;
    public float Z;

    public enum Axis
    {
        X,
        Y,
        Z
    }

    public Vector3(float x, float y, float z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public float this[int index]
    {
        readonly get
        {
            return index switch
            {
                0 => X,
                1 => Y,
                2 => Z,
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
                case 2:
                    Z = value;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
    }

    private static readonly Vector3 _zero = new Vector3(0f, 0f, 0f);
    private static readonly Vector3 _one = new Vector3(1f, 1f, 1f);
    private static readonly Vector3 _inf = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
    private static readonly Vector3 _up = new Vector3(0f, 1f, 0f);
    private static readonly Vector3 _down = new Vector3(0f, -1f, 0f);
    private static readonly Vector3 _right = new Vector3(1f, 0f, 0f);
    private static readonly Vector3 _left = new Vector3(-1f, 0f, 0f);
    private static readonly Vector3 _forward = new Vector3(0f, 0f, -1f);
    private static readonly Vector3 _back = new Vector3(0f, 0f, 1f);
    private static readonly Vector3 _modelLeft = new Vector3(1f, 0f, 0f);
    private static readonly Vector3 _modelRight = new Vector3(-1f, 0f, 0f);
    private static readonly Vector3 _modelTop = new Vector3(0f, 1f, 0f);
    private static readonly Vector3 _modelBottom = new Vector3(0f, -1f, 0f);
    private static readonly Vector3 _modelFront = new Vector3(0f, 0f, 1f);
    private static readonly Vector3 _modelRear = new Vector3(0f, 0f, -1f);

    public static Vector3 Zero => _zero;
    public static Vector3 One => _one;
    public static Vector3 Inf => _inf;
    public static Vector3 Up => _up;
    public static Vector3 Down => _down;
    public static Vector3 Right => _right;
    public static Vector3 Left => _left;
    public static Vector3 Forward => _forward;
    public static Vector3 Back => _back;
    public static Vector3 ModelLeft => _modelLeft;
    public static Vector3 ModelRight => _modelRight;
    public static Vector3 ModelTop => _modelTop;
    public static Vector3 ModelBottom => _modelBottom;
    public static Vector3 ModelFront => _modelFront;
    public static Vector3 ModelRear => _modelRear;

    public readonly void Deconstruct(out float x, out float y, out float z)
    {
        x = X;
        y = Y;
        z = Z;
    }

    internal void Normalize()
    {
        float lengthSquared = LengthSquared();
        if (lengthSquared == 0f)
        {
            X = 0f;
            Y = 0f;
            Z = 0f;
        }
        else
        {
            float length = Mathf.Sqrt(lengthSquared);
            X /= length;
            Y /= length;
            Z /= length;
        }
    }

    public readonly Vector3 Abs() => new Vector3(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z));

    public readonly Vector3 Ceil() => new Vector3(Mathf.Ceil(X), Mathf.Ceil(Y), Mathf.Ceil(Z));

    public readonly Vector3 Floor() => new Vector3(Mathf.Floor(X), Mathf.Floor(Y), Mathf.Floor(Z));

    public readonly Vector3 Round() => new Vector3(Mathf.Round(X), Mathf.Round(Y), Mathf.Round(Z));

    public readonly Vector3 Sign() => new Vector3(Mathf.Sign(X), Mathf.Sign(Y), Mathf.Sign(Z));

    public readonly float AngleTo(Vector3 to) => Mathf.Atan2(Cross(to).Length(), Dot(to));

    public readonly Vector3 Bounce(Vector3 normal) => -Reflect(normal);

    public readonly Vector3 Clamp(Vector3 min, Vector3 max)
    {
        return new Vector3(
            Mathf.Clamp(X, min.X, max.X),
            Mathf.Clamp(Y, min.Y, max.Y),
            Mathf.Clamp(Z, min.Z, max.Z));
    }

    public readonly Vector3 Clamp(float min, float max)
    {
        return new Vector3(
            Mathf.Clamp(X, min, max),
            Mathf.Clamp(Y, min, max),
            Mathf.Clamp(Z, min, max));
    }

    public readonly Vector3 Cross(Vector3 with)
    {
        return new Vector3(
            Y * with.Z - Z * with.Y,
            Z * with.X - X * with.Z,
            X * with.Y - Y * with.X);
    }

    public readonly Vector3 CubicInterpolate(Vector3 b, Vector3 preA, Vector3 postB, float weight)
    {
        return new Vector3(
            Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight),
            Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight),
            Mathf.CubicInterpolate(Z, b.Z, preA.Z, postB.Z, weight));
    }

    public readonly Vector3 CubicInterpolateInTime(Vector3 b, Vector3 preA, Vector3 postB, float weight, float t, float preAT, float postBT)
    {
        return new Vector3(
            Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT),
            Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT),
            Mathf.CubicInterpolateInTime(Z, b.Z, preA.Z, postB.Z, weight, t, preAT, postBT));
    }

    public readonly Vector3 BezierInterpolate(Vector3 control1, Vector3 control2, Vector3 end, float t)
    {
        return new Vector3(
            Mathf.BezierInterpolate(X, control1.X, control2.X, end.X, t),
            Mathf.BezierInterpolate(Y, control1.Y, control2.Y, end.Y, t),
            Mathf.BezierInterpolate(Z, control1.Z, control2.Z, end.Z, t));
    }

    public readonly Vector3 BezierDerivative(Vector3 control1, Vector3 control2, Vector3 end, float t)
    {
        return new Vector3(
            Mathf.BezierDerivative(X, control1.X, control2.X, end.X, t),
            Mathf.BezierDerivative(Y, control1.Y, control2.Y, end.Y, t),
            Mathf.BezierDerivative(Z, control1.Z, control2.Z, end.Z, t));
    }

    public readonly Vector3 DirectionTo(Vector3 to) => (to - this).Normalized();

    public readonly float DistanceSquaredTo(Vector3 to) => (to - this).LengthSquared();

    public readonly float DistanceTo(Vector3 to) => (to - this).Length();

    public readonly float Dot(Vector3 with) => X * with.X + Y * with.Y + Z * with.Z;

    public readonly Vector3 Inverse() => new Vector3(1f / X, 1f / Y, 1f / Z);

    public readonly bool IsFinite() => Mathf.IsFinite(X) && Mathf.IsFinite(Y) && Mathf.IsFinite(Z);

    public readonly bool IsNormalized() => Mathf.Abs(LengthSquared() - 1f) < 1e-6f;

    public readonly float Length() => Mathf.Sqrt(X * X + Y * Y + Z * Z);

    public readonly float LengthSquared() => X * X + Y * Y + Z * Z;

    public readonly Vector3 Lerp(Vector3 to, float weight)
    {
        return new Vector3(
            Mathf.Lerp(X, to.X, weight),
            Mathf.Lerp(Y, to.Y, weight),
            Mathf.Lerp(Z, to.Z, weight));
    }

    public readonly Vector3 LimitLength(float length = 1f)
    {
        float currentLength = Length();
        Vector3 result = this;
        if (currentLength > 0f && length < currentLength)
        {
            result /= currentLength;
            result *= length;
        }

        return result;
    }

    public readonly Vector3 Max(Vector3 with)
    {
        return new Vector3(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y), Mathf.Max(Z, with.Z));
    }

    public readonly Vector3 Max(float with)
    {
        return new Vector3(Mathf.Max(X, with), Mathf.Max(Y, with), Mathf.Max(Z, with));
    }

    public readonly Vector3 Min(Vector3 with)
    {
        return new Vector3(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y), Mathf.Min(Z, with.Z));
    }

    public readonly Vector3 Min(float with)
    {
        return new Vector3(Mathf.Min(X, with), Mathf.Min(Y, with), Mathf.Min(Z, with));
    }

    public readonly Axis MaxAxisIndex()
    {
        if (X < Y)
        {
            return Y < Z ? Axis.Z : Axis.Y;
        }

        return X < Z ? Axis.Z : Axis.X;
    }

    public readonly Axis MinAxisIndex()
    {
        if (X < Y)
        {
            return X < Z ? Axis.X : Axis.Z;
        }

        return Y < Z ? Axis.Y : Axis.Z;
    }

    public readonly Vector3 MoveToward(Vector3 to, float delta)
    {
        Vector3 offset = to - this;
        float distance = offset.Length();
        if (distance <= delta || distance < 1e-6f)
        {
            return to;
        }

        return this + offset / distance * delta;
    }

    public readonly Vector3 Normalized()
    {
        Vector3 result = this;
        result.Normalize();
        return result;
    }

    public readonly object Outer(Vector3 with)
    {
        throw new NotSupportedException("Outer requires Basis, which the headless shim does not provide.");
    }

    public readonly Vector3 PosMod(float mod)
    {
        return new Vector3(Mathf.PosMod(X, mod), Mathf.PosMod(Y, mod), Mathf.PosMod(Z, mod));
    }

    public readonly Vector3 PosMod(Vector3 modv)
    {
        return new Vector3(Mathf.PosMod(X, modv.X), Mathf.PosMod(Y, modv.Y), Mathf.PosMod(Z, modv.Z));
    }

    public readonly Vector3 Project(Vector3 onNormal)
    {
        return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
    }

    public readonly Vector3 Reflect(Vector3 normal)
    {
        return 2f * Dot(normal) * normal - this;
    }

    public readonly Vector3 Rotated(Vector3 axis, float angle)
    {
        throw new NotSupportedException("Rotated requires Basis, which the headless shim does not provide.");
    }

    public readonly float SignedAngleTo(Vector3 to, Vector3 axis)
    {
        Vector3 crossTo = Cross(to);
        float unsignedAngle = Mathf.Atan2(crossTo.Length(), Dot(to));
        return crossTo.Dot(axis) < 0f ? -unsignedAngle : unsignedAngle;
    }

    public readonly Vector3 Slerp(Vector3 to, float weight)
    {
        float startLengthSquared = LengthSquared();
        float endLengthSquared = to.LengthSquared();
        if (startLengthSquared == 0f || endLengthSquared == 0f)
        {
            return Lerp(to, weight);
        }

        Vector3 axis = Cross(to);
        float axisLengthSquared = axis.LengthSquared();
        if (axisLengthSquared == 0f)
        {
            return Lerp(to, weight);
        }

        axis /= Mathf.Sqrt(axisLengthSquared);
        float startLength = Mathf.Sqrt(startLengthSquared);
        float resultLength = Mathf.Lerp(startLength, Mathf.Sqrt(endLengthSquared), weight);
        float angle = AngleTo(to);
        return Rotated(axis, angle * weight) * (resultLength / startLength);
    }

    public readonly Vector3 Slide(Vector3 normal) => this - normal * Dot(normal);

    public readonly Vector3 Snapped(Vector3 step)
    {
        return new Vector3(
            Mathf.Snapped(X, step.X),
            Mathf.Snapped(Y, step.Y),
            Mathf.Snapped(Z, step.Z));
    }

    public readonly Vector3 Snapped(float step)
    {
        return new Vector3(
            Mathf.Snapped(X, step),
            Mathf.Snapped(Y, step),
            Mathf.Snapped(Z, step));
    }

    public readonly Vector2 OctahedronEncode()
    {
        Vector3 normal = this / (Mathf.Abs(X) + Mathf.Abs(Y) + Mathf.Abs(Z));
        Vector2 result;
        if (normal.Z >= 0f)
        {
            result = new Vector2(normal.X, normal.Y);
        }
        else
        {
            result = new Vector2(
                (1f - Mathf.Abs(normal.Y)) * SignNonNegative(normal.X),
                (1f - Mathf.Abs(normal.X)) * SignNonNegative(normal.Y));
        }

        result = result * 0.5f + new Vector2(0.5f, 0.5f);
        return result;
    }

    public static Vector3 OctahedronDecode(Vector2 oct)
    {
        Vector2 f = oct * 2f - new Vector2(1f, 1f);
        Vector3 normal = new Vector3(f.X, f.Y, 1f - Mathf.Abs(f.X) - Mathf.Abs(f.Y));
        float t = Mathf.Clamp(-normal.Z, 0f, 1f);
        normal.X += normal.X >= 0f ? -t : t;
        normal.Y += normal.Y >= 0f ? -t : t;
        return normal.Normalized();
    }

    internal readonly Vector3 GetAnyPerpendicular()
    {
        if (IsZeroApprox())
        {
            throw new ArgumentException("The vector must not be zero.");
        }

        Vector3 pick = Mathf.Abs(X) <= Mathf.Abs(Y) && Mathf.Abs(X) <= Mathf.Abs(Z)
            ? new Vector3(1f, 0f, 0f)
            : new Vector3(0f, 1f, 0f);
        return Cross(pick).Normalized();
    }

    private static float SignNonNegative(float value) => value >= 0f ? 1f : -1f;

    public readonly bool IsEqualApprox(Vector3 other)
    {
        return Mathf.IsEqualApprox(X, other.X)
            && Mathf.IsEqualApprox(Y, other.Y)
            && Mathf.IsEqualApprox(Z, other.Z);
    }

    public readonly bool IsZeroApprox()
    {
        return Mathf.IsZeroApprox(X) && Mathf.IsZeroApprox(Y) && Mathf.IsZeroApprox(Z);
    }

    public readonly bool Equals(Vector3 other) => X == other.X && Y == other.Y && Z == other.Z;

    public override readonly bool Equals(object? obj) => obj is Vector3 other && Equals(other);

    public override readonly int GetHashCode() => HashCode.Combine(X, Y, Z);

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)})";
    }

    public static Vector3 operator +(Vector3 left, Vector3 right)
    {
        return new Vector3(left.X + right.X, left.Y + right.Y, left.Z + right.Z);
    }

    public static Vector3 operator -(Vector3 left, Vector3 right)
    {
        return new Vector3(left.X - right.X, left.Y - right.Y, left.Z - right.Z);
    }

    public static Vector3 operator -(Vector3 vec) => new Vector3(-vec.X, -vec.Y, -vec.Z);

    public static Vector3 operator *(Vector3 vec, float scale)
    {
        return new Vector3(vec.X * scale, vec.Y * scale, vec.Z * scale);
    }

    public static Vector3 operator *(float scale, Vector3 vec)
    {
        return new Vector3(vec.X * scale, vec.Y * scale, vec.Z * scale);
    }

    public static Vector3 operator *(Vector3 left, Vector3 right)
    {
        return new Vector3(left.X * right.X, left.Y * right.Y, left.Z * right.Z);
    }

    public static Vector3 operator /(Vector3 vec, float divisor)
    {
        return new Vector3(vec.X / divisor, vec.Y / divisor, vec.Z / divisor);
    }

    public static Vector3 operator /(Vector3 vec, Vector3 divisorv)
    {
        return new Vector3(vec.X / divisorv.X, vec.Y / divisorv.Y, vec.Z / divisorv.Z);
    }

    public static Vector3 operator %(Vector3 vec, float divisor)
    {
        return new Vector3(vec.X % divisor, vec.Y % divisor, vec.Z % divisor);
    }

    public static Vector3 operator %(Vector3 vec, Vector3 divisorv)
    {
        return new Vector3(vec.X % divisorv.X, vec.Y % divisorv.Y, vec.Z % divisorv.Z);
    }

    public static bool operator ==(Vector3 left, Vector3 right) => left.Equals(right);

    public static bool operator !=(Vector3 left, Vector3 right) => !left.Equals(right);

    public static bool operator <(Vector3 left, Vector3 right)
    {
        if (left.X == right.X)
        {
            return left.Y == right.Y ? left.Z < right.Z : left.Y < right.Y;
        }

        return left.X < right.X;
    }

    public static bool operator >(Vector3 left, Vector3 right)
    {
        if (left.X == right.X)
        {
            return left.Y == right.Y ? left.Z > right.Z : left.Y > right.Y;
        }

        return left.X > right.X;
    }

    public static bool operator <=(Vector3 left, Vector3 right)
    {
        if (left.X == right.X)
        {
            return left.Y == right.Y ? left.Z <= right.Z : left.Y < right.Y;
        }

        return left.X < right.X;
    }

    public static bool operator >=(Vector3 left, Vector3 right)
    {
        if (left.X == right.X)
        {
            return left.Y == right.Y ? left.Z >= right.Z : left.Y > right.Y;
        }

        return left.X > right.X;
    }
}
