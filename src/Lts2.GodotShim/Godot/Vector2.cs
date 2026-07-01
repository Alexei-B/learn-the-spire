using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Godot;

[Serializable]
public struct Vector2 : IEquatable<Vector2>
{
	public enum Axis
	{
		X,
		Y
	}

	public float X;

	public float Y;

	private static readonly Vector2 _zero = new Vector2(0f, 0f);

	private static readonly Vector2 _one = new Vector2(1f, 1f);

	private static readonly Vector2 _inf = new Vector2(float.PositiveInfinity, float.PositiveInfinity);

	private static readonly Vector2 _up = new Vector2(0f, -1f);

	private static readonly Vector2 _down = new Vector2(0f, 1f);

	private static readonly Vector2 _right = new Vector2(1f, 0f);

	private static readonly Vector2 _left = new Vector2(-1f, 0f);

	public float this[int index]
	{
		readonly get
		{
			return index switch
			{
				0 => X, 
				1 => Y, 
				_ => throw new ArgumentOutOfRangeException("index"), 
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
				throw new ArgumentOutOfRangeException("index");
			}
		}
	}

	public static Vector2 Zero => _zero;

	public static Vector2 One => _one;

	public static Vector2 Inf => _inf;

	public static Vector2 Up => _up;

	public static Vector2 Down => _down;

	public static Vector2 Right => _right;

	public static Vector2 Left => _left;

	public readonly void Deconstruct(out float x, out float y)
	{
		x = X;
		y = Y;
	}

	internal void Normalize()
	{
		float num = LengthSquared();
		if (num == 0f)
		{
			X = (Y = 0f);
			return;
		}
		float num2 = Mathf.Sqrt(num);
		X /= num2;
		Y /= num2;
	}

	public readonly Vector2 Abs()
	{
		return new Vector2(Mathf.Abs(X), Mathf.Abs(Y));
	}

	public readonly float Angle()
	{
		return Mathf.Atan2(Y, X);
	}

	public readonly float AngleTo(Vector2 to)
	{
		return Mathf.Atan2(Cross(to), Dot(to));
	}

	public readonly float AngleToPoint(Vector2 to)
	{
		return Mathf.Atan2(to.Y - Y, to.X - X);
	}

	public readonly float Aspect()
	{
		return X / Y;
	}

	public readonly Vector2 Bounce(Vector2 normal)
	{
		return -Reflect(normal);
	}

	public readonly Vector2 Ceil()
	{
		return new Vector2(Mathf.Ceil(X), Mathf.Ceil(Y));
	}

	public readonly Vector2 Clamp(Vector2 min, Vector2 max)
	{
		return new Vector2(Mathf.Clamp(X, min.X, max.X), Mathf.Clamp(Y, min.Y, max.Y));
	}

	public readonly Vector2 Clamp(float min, float max)
	{
		return new Vector2(Mathf.Clamp(X, min, max), Mathf.Clamp(Y, min, max));
	}

	public readonly float Cross(Vector2 with)
	{
		return X * with.Y - Y * with.X;
	}

	public readonly Vector2 CubicInterpolate(Vector2 b, Vector2 preA, Vector2 postB, float weight)
	{
		return new Vector2(Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight), Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight));
	}

	public readonly Vector2 CubicInterpolateInTime(Vector2 b, Vector2 preA, Vector2 postB, float weight, float t, float preAT, float postBT)
	{
		return new Vector2(Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT), Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT));
	}

	public readonly Vector2 BezierInterpolate(Vector2 control1, Vector2 control2, Vector2 end, float t)
	{
		return new Vector2(Mathf.BezierInterpolate(X, control1.X, control2.X, end.X, t), Mathf.BezierInterpolate(Y, control1.Y, control2.Y, end.Y, t));
	}

	public readonly Vector2 BezierDerivative(Vector2 control1, Vector2 control2, Vector2 end, float t)
	{
		return new Vector2(Mathf.BezierDerivative(X, control1.X, control2.X, end.X, t), Mathf.BezierDerivative(Y, control1.Y, control2.Y, end.Y, t));
	}

	public readonly Vector2 DirectionTo(Vector2 to)
	{
		return new Vector2(to.X - X, to.Y - Y).Normalized();
	}

	public readonly float DistanceSquaredTo(Vector2 to)
	{
		return (X - to.X) * (X - to.X) + (Y - to.Y) * (Y - to.Y);
	}

	public readonly float DistanceTo(Vector2 to)
	{
		return Mathf.Sqrt((X - to.X) * (X - to.X) + (Y - to.Y) * (Y - to.Y));
	}

	public readonly float Dot(Vector2 with)
	{
		return X * with.X + Y * with.Y;
	}

	public readonly Vector2 Floor()
	{
		return new Vector2(Mathf.Floor(X), Mathf.Floor(Y));
	}

	public readonly Vector2 Inverse()
	{
		return new Vector2(1f / X, 1f / Y);
	}

	public readonly bool IsFinite()
	{
		if (Mathf.IsFinite(X))
		{
			return Mathf.IsFinite(Y);
		}
		return false;
	}

	public readonly bool IsNormalized()
	{
		return Mathf.Abs(LengthSquared() - 1f) < 1E-06f;
	}

	public readonly float Length()
	{
		return Mathf.Sqrt(X * X + Y * Y);
	}

	public readonly float LengthSquared()
	{
		return X * X + Y * Y;
	}

	public readonly Vector2 Lerp(Vector2 to, float weight)
	{
		return new Vector2(Mathf.Lerp(X, to.X, weight), Mathf.Lerp(Y, to.Y, weight));
	}

	public readonly Vector2 LimitLength(float length = 1f)
	{
		Vector2 result = this;
		float num = Length();
		if (num > 0f && length < num)
		{
			result /= num;
			result *= length;
		}
		return result;
	}

	public readonly Vector2 Max(Vector2 with)
	{
		return new Vector2(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y));
	}

	public readonly Vector2 Max(float with)
	{
		return new Vector2(Mathf.Max(X, with), Mathf.Max(Y, with));
	}

	public readonly Vector2 Min(Vector2 with)
	{
		return new Vector2(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y));
	}

	public readonly Vector2 Min(float with)
	{
		return new Vector2(Mathf.Min(X, with), Mathf.Min(Y, with));
	}

	public readonly Axis MaxAxisIndex()
	{
		if (!(X < Y))
		{
			return Axis.X;
		}
		return Axis.Y;
	}

	public readonly Axis MinAxisIndex()
	{
		if (!(X < Y))
		{
			return Axis.Y;
		}
		return Axis.X;
	}

	public readonly Vector2 MoveToward(Vector2 to, float delta)
	{
		Vector2 vector = this;
		Vector2 vector2 = to - vector;
		float num = vector2.Length();
		if (num <= delta || num < 1E-06f)
		{
			return to;
		}
		return vector + vector2 / num * delta;
	}

	public readonly Vector2 Normalized()
	{
		Vector2 result = this;
		result.Normalize();
		return result;
	}

	public readonly Vector2 PosMod(float mod)
	{
		Vector2 result = default(Vector2);
		result.X = Mathf.PosMod(X, mod);
		result.Y = Mathf.PosMod(Y, mod);
		return result;
	}

	public readonly Vector2 PosMod(Vector2 modv)
	{
		Vector2 result = default(Vector2);
		result.X = Mathf.PosMod(X, modv.X);
		result.Y = Mathf.PosMod(Y, modv.Y);
		return result;
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
		var (num, num2) = Mathf.SinCos(angle);
		return new Vector2(X * num2 - Y * num, X * num + Y * num2);
	}

	public readonly Vector2 Round()
	{
		return new Vector2(Mathf.Round(X), Mathf.Round(Y));
	}

	public readonly Vector2 Sign()
	{
		Vector2 result = default(Vector2);
		result.X = Mathf.Sign(X);
		result.Y = Mathf.Sign(Y);
		return result;
	}

	public readonly Vector2 Slerp(Vector2 to, float weight)
	{
		float num = LengthSquared();
		float num2 = to.LengthSquared();
		if ((double)num == 0.0 || (double)num2 == 0.0)
		{
			return Lerp(to, weight);
		}
		float num3 = Mathf.Sqrt(num);
		float num4 = Mathf.Lerp(num3, Mathf.Sqrt(num2), weight);
		float num5 = AngleTo(to);
		return Rotated(num5 * weight) * (num4 / num3);
	}

	public readonly Vector2 Slide(Vector2 normal)
	{
		return this - normal * Dot(normal);
	}

	public readonly Vector2 Snapped(Vector2 step)
	{
		return new Vector2(Mathf.Snapped(X, step.X), Mathf.Snapped(Y, step.Y));
	}

	public readonly Vector2 Snapped(float step)
	{
		return new Vector2(Mathf.Snapped(X, step), Mathf.Snapped(Y, step));
	}

	public readonly Vector2 Orthogonal()
	{
		return new Vector2(Y, 0f - X);
	}

	public Vector2(float x, float y)
	{
		X = x;
		Y = y;
	}

	public static Vector2 FromAngle(float angle)
	{
		(float Sin, float Cos) tuple = Mathf.SinCos(angle);
		var (y, _) = tuple;
		return new Vector2(tuple.Cos, y);
	}

	public static Vector2 operator +(Vector2 left, Vector2 right)
	{
		left.X += right.X;
		left.Y += right.Y;
		return left;
	}

	public static Vector2 operator -(Vector2 left, Vector2 right)
	{
		left.X -= right.X;
		left.Y -= right.Y;
		return left;
	}

	public static Vector2 operator -(Vector2 vec)
	{
		vec.X = 0f - vec.X;
		vec.Y = 0f - vec.Y;
		return vec;
	}

	public static Vector2 operator *(Vector2 vec, float scale)
	{
		vec.X *= scale;
		vec.Y *= scale;
		return vec;
	}

	public static Vector2 operator *(float scale, Vector2 vec)
	{
		vec.X *= scale;
		vec.Y *= scale;
		return vec;
	}

	public static Vector2 operator *(Vector2 left, Vector2 right)
	{
		left.X *= right.X;
		left.Y *= right.Y;
		return left;
	}

	public static Vector2 operator /(Vector2 vec, float divisor)
	{
		vec.X /= divisor;
		vec.Y /= divisor;
		return vec;
	}

	public static Vector2 operator /(Vector2 vec, Vector2 divisorv)
	{
		vec.X /= divisorv.X;
		vec.Y /= divisorv.Y;
		return vec;
	}

	public static Vector2 operator %(Vector2 vec, float divisor)
	{
		vec.X %= divisor;
		vec.Y %= divisor;
		return vec;
	}

	public static Vector2 operator %(Vector2 vec, Vector2 divisorv)
	{
		vec.X %= divisorv.X;
		vec.Y %= divisorv.Y;
		return vec;
	}

	public static bool operator ==(Vector2 left, Vector2 right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Vector2 left, Vector2 right)
	{
		return !left.Equals(right);
	}

	public static bool operator <(Vector2 left, Vector2 right)
	{
		if (left.X == right.X)
		{
			return left.Y < right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >(Vector2 left, Vector2 right)
	{
		if (left.X == right.X)
		{
			return left.Y > right.Y;
		}
		return left.X > right.X;
	}

	public static bool operator <=(Vector2 left, Vector2 right)
	{
		if (left.X == right.X)
		{
			return left.Y <= right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >=(Vector2 left, Vector2 right)
	{
		if (left.X == right.X)
		{
			return left.Y >= right.Y;
		}
		return left.X > right.X;
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Vector2 other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Vector2 other)
	{
		if (X == other.X)
		{
			return Y == other.Y;
		}
		return false;
	}

	public readonly bool IsEqualApprox(Vector2 other)
	{
		if (Mathf.IsEqualApprox(X, other.X))
		{
			return Mathf.IsEqualApprox(Y, other.Y);
		}
		return false;
	}

	public readonly bool IsZeroApprox()
	{
		if (Mathf.IsZeroApprox(X))
		{
			return Mathf.IsZeroApprox(Y);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(X, Y);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)})";
	}
}
