using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Godot;

// Value type copied verbatim from refsrc/GodotSharp (pure managed, no native calls), so its
// arithmetic/value semantics stay faithful. The only deviation: the two methods that return/use
// the 3x3 Basis type (Outer, Rotated) throw instead — Basis (and its Quaternion dependency) are
// not part of the shim, and headless code never takes the 3D-rotation paths that need them.
[Serializable]
public struct Vector3 : IEquatable<Vector3>
{
	public enum Axis
	{
		X,
		Y,
		Z
	}

	public float X;

	public float Y;

	public float Z;

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

	public float this[int index]
	{
		readonly get
		{
			return index switch
			{
				0 => X,
				1 => Y,
				2 => Z,
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
			case 2:
				Z = value;
				break;
			default:
				throw new ArgumentOutOfRangeException("index");
			}
		}
	}

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
		float num = LengthSquared();
		if (num == 0f)
		{
			X = (Y = (Z = 0f));
			return;
		}
		float num2 = Mathf.Sqrt(num);
		X /= num2;
		Y /= num2;
		Z /= num2;
	}

	public readonly Vector3 Abs()
	{
		return new Vector3(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z));
	}

	public readonly float AngleTo(Vector3 to)
	{
		return Mathf.Atan2(Cross(to).Length(), Dot(to));
	}

	public readonly Vector3 Bounce(Vector3 normal)
	{
		return -Reflect(normal);
	}

	public readonly Vector3 Ceil()
	{
		return new Vector3(Mathf.Ceil(X), Mathf.Ceil(Y), Mathf.Ceil(Z));
	}

	public readonly Vector3 Clamp(Vector3 min, Vector3 max)
	{
		return new Vector3(Mathf.Clamp(X, min.X, max.X), Mathf.Clamp(Y, min.Y, max.Y), Mathf.Clamp(Z, min.Z, max.Z));
	}

	public readonly Vector3 Clamp(float min, float max)
	{
		return new Vector3(Mathf.Clamp(X, min, max), Mathf.Clamp(Y, min, max), Mathf.Clamp(Z, min, max));
	}

	public readonly Vector3 Cross(Vector3 with)
	{
		return new Vector3(Y * with.Z - Z * with.Y, Z * with.X - X * with.Z, X * with.Y - Y * with.X);
	}

	public readonly Vector3 CubicInterpolate(Vector3 b, Vector3 preA, Vector3 postB, float weight)
	{
		return new Vector3(Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight), Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight), Mathf.CubicInterpolate(Z, b.Z, preA.Z, postB.Z, weight));
	}

	public readonly Vector3 CubicInterpolateInTime(Vector3 b, Vector3 preA, Vector3 postB, float weight, float t, float preAT, float postBT)
	{
		return new Vector3(Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT), Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT), Mathf.CubicInterpolateInTime(Z, b.Z, preA.Z, postB.Z, weight, t, preAT, postBT));
	}

	public readonly Vector3 BezierInterpolate(Vector3 control1, Vector3 control2, Vector3 end, float t)
	{
		return new Vector3(Mathf.BezierInterpolate(X, control1.X, control2.X, end.X, t), Mathf.BezierInterpolate(Y, control1.Y, control2.Y, end.Y, t), Mathf.BezierInterpolate(Z, control1.Z, control2.Z, end.Z, t));
	}

	public readonly Vector3 BezierDerivative(Vector3 control1, Vector3 control2, Vector3 end, float t)
	{
		return new Vector3(Mathf.BezierDerivative(X, control1.X, control2.X, end.X, t), Mathf.BezierDerivative(Y, control1.Y, control2.Y, end.Y, t), Mathf.BezierDerivative(Z, control1.Z, control2.Z, end.Z, t));
	}

	public readonly Vector3 DirectionTo(Vector3 to)
	{
		return new Vector3(to.X - X, to.Y - Y, to.Z - Z).Normalized();
	}

	public readonly float DistanceSquaredTo(Vector3 to)
	{
		return (to - this).LengthSquared();
	}

	public readonly float DistanceTo(Vector3 to)
	{
		return (to - this).Length();
	}

	public readonly float Dot(Vector3 with)
	{
		return X * with.X + Y * with.Y + Z * with.Z;
	}

	public readonly Vector3 Floor()
	{
		return new Vector3(Mathf.Floor(X), Mathf.Floor(Y), Mathf.Floor(Z));
	}

	public readonly Vector3 Inverse()
	{
		return new Vector3(1f / X, 1f / Y, 1f / Z);
	}

	public readonly bool IsFinite()
	{
		if (Mathf.IsFinite(X) && Mathf.IsFinite(Y))
		{
			return Mathf.IsFinite(Z);
		}
		return false;
	}

	public readonly bool IsNormalized()
	{
		return Mathf.Abs(LengthSquared() - 1f) < 1E-06f;
	}

	public readonly float Length()
	{
		float num = X * X;
		float num2 = Y * Y;
		float num3 = Z * Z;
		return Mathf.Sqrt(num + num2 + num3);
	}

	public readonly float LengthSquared()
	{
		float num = X * X;
		float num2 = Y * Y;
		float num3 = Z * Z;
		return num + num2 + num3;
	}

	public readonly Vector3 Lerp(Vector3 to, float weight)
	{
		return new Vector3(Mathf.Lerp(X, to.X, weight), Mathf.Lerp(Y, to.Y, weight), Mathf.Lerp(Z, to.Z, weight));
	}

	public readonly Vector3 LimitLength(float length = 1f)
	{
		Vector3 result = this;
		float num = Length();
		if (num > 0f && length < num)
		{
			result /= num;
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
		if (!(X < Y))
		{
			if (!(X < Z))
			{
				return Axis.X;
			}
			return Axis.Z;
		}
		if (!(Y < Z))
		{
			return Axis.Y;
		}
		return Axis.Z;
	}

	public readonly Axis MinAxisIndex()
	{
		if (!(X < Y))
		{
			if (!(Y < Z))
			{
				return Axis.Z;
			}
			return Axis.Y;
		}
		if (!(X < Z))
		{
			return Axis.Z;
		}
		return Axis.X;
	}

	public readonly Vector3 MoveToward(Vector3 to, float delta)
	{
		Vector3 vector = this;
		Vector3 vector2 = to - vector;
		float num = vector2.Length();
		if (num <= delta || num < 1E-06f)
		{
			return to;
		}
		return vector + vector2 / num * delta;
	}

	public readonly Vector3 Normalized()
	{
		Vector3 result = this;
		result.Normalize();
		return result;
	}

	// Returns a Basis (3x3 matrix) in real Godot; Basis is not part of the headless shim.
	public readonly object Outer(Vector3 with)
	{
		throw new NotSupportedException("Vector3.Outer requires Basis, which the headless shim omits.");
	}

	public readonly Vector3 PosMod(float mod)
	{
		Vector3 result = default(Vector3);
		result.X = Mathf.PosMod(X, mod);
		result.Y = Mathf.PosMod(Y, mod);
		result.Z = Mathf.PosMod(Z, mod);
		return result;
	}

	public readonly Vector3 PosMod(Vector3 modv)
	{
		Vector3 result = default(Vector3);
		result.X = Mathf.PosMod(X, modv.X);
		result.Y = Mathf.PosMod(Y, modv.Y);
		result.Z = Mathf.PosMod(Z, modv.Z);
		return result;
	}

	public readonly Vector3 Project(Vector3 onNormal)
	{
		return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
	}

	public readonly Vector3 Reflect(Vector3 normal)
	{
		return 2f * Dot(normal) * normal - this;
	}

	// Rotates around an axis via a Basis in real Godot; Basis is not part of the headless shim.
	public readonly Vector3 Rotated(Vector3 axis, float angle)
	{
		throw new NotSupportedException("Vector3.Rotated requires Basis, which the headless shim omits.");
	}

	public readonly Vector3 Round()
	{
		return new Vector3(Mathf.Round(X), Mathf.Round(Y), Mathf.Round(Z));
	}

	public readonly Vector3 Sign()
	{
		Vector3 result = default(Vector3);
		result.X = Mathf.Sign(X);
		result.Y = Mathf.Sign(Y);
		result.Z = Mathf.Sign(Z);
		return result;
	}

	public readonly float SignedAngleTo(Vector3 to, Vector3 axis)
	{
		Vector3 vector = Cross(to);
		float num = Mathf.Atan2(vector.Length(), Dot(to));
		if (!(vector.Dot(axis) < 0f))
		{
			return num;
		}
		return 0f - num;
	}

	public readonly Vector3 Slerp(Vector3 to, float weight)
	{
		float num = LengthSquared();
		float num2 = to.LengthSquared();
		if ((double)num == 0.0 || (double)num2 == 0.0)
		{
			return Lerp(to, weight);
		}
		Vector3 axis = Cross(to);
		float num3 = axis.LengthSquared();
		if ((double)num3 == 0.0)
		{
			return Lerp(to, weight);
		}
		axis /= Mathf.Sqrt(num3);
		float num4 = Mathf.Sqrt(num);
		float num5 = Mathf.Lerp(num4, Mathf.Sqrt(num2), weight);
		float num6 = AngleTo(to);
		return Rotated(axis, num6 * weight) * (num5 / num4);
	}

	public readonly Vector3 Slide(Vector3 normal)
	{
		return this - normal * Dot(normal);
	}

	public readonly Vector3 Snapped(Vector3 step)
	{
		return new Vector3(Mathf.Snapped(X, step.X), Mathf.Snapped(Y, step.Y), Mathf.Snapped(Z, step.Z));
	}

	public readonly Vector3 Snapped(float step)
	{
		return new Vector3(Mathf.Snapped(X, step), Mathf.Snapped(Y, step), Mathf.Snapped(Z, step));
	}

	public readonly Vector2 OctahedronEncode()
	{
		Vector3 vector = this;
		vector /= Mathf.Abs(vector.X) + Mathf.Abs(vector.Y) + Mathf.Abs(vector.Z);
		Vector2 result = default(Vector2);
		if (vector.Z >= 0f)
		{
			result.X = vector.X;
			result.Y = vector.Y;
		}
		else
		{
			result.X = (1f - Mathf.Abs(vector.Y)) * ((vector.X >= 0f) ? 1f : (-1f));
			result.Y = (1f - Mathf.Abs(vector.X)) * ((vector.Y >= 0f) ? 1f : (-1f));
		}
		result.X = result.X * 0.5f + 0.5f;
		result.Y = result.Y * 0.5f + 0.5f;
		return result;
	}

	public static Vector3 OctahedronDecode(Vector2 oct)
	{
		Vector2 vector = new Vector2(oct.X * 2f - 1f, oct.Y * 2f - 1f);
		Vector3 vector2 = new Vector3(vector.X, vector.Y, 1f - Mathf.Abs(vector.X) - Mathf.Abs(vector.Y));
		float num = Mathf.Clamp(0f - vector2.Z, 0f, 1f);
		vector2.X += ((vector2.X >= 0f) ? (0f - num) : num);
		vector2.Y += ((vector2.Y >= 0f) ? (0f - num) : num);
		return vector2.Normalized();
	}

	public Vector3(float x, float y, float z)
	{
		X = x;
		Y = y;
		Z = z;
	}

	public static Vector3 operator +(Vector3 left, Vector3 right)
	{
		left.X += right.X;
		left.Y += right.Y;
		left.Z += right.Z;
		return left;
	}

	public static Vector3 operator -(Vector3 left, Vector3 right)
	{
		left.X -= right.X;
		left.Y -= right.Y;
		left.Z -= right.Z;
		return left;
	}

	public static Vector3 operator -(Vector3 vec)
	{
		vec.X = 0f - vec.X;
		vec.Y = 0f - vec.Y;
		vec.Z = 0f - vec.Z;
		return vec;
	}

	public static Vector3 operator *(Vector3 vec, float scale)
	{
		vec.X *= scale;
		vec.Y *= scale;
		vec.Z *= scale;
		return vec;
	}

	public static Vector3 operator *(float scale, Vector3 vec)
	{
		vec.X *= scale;
		vec.Y *= scale;
		vec.Z *= scale;
		return vec;
	}

	public static Vector3 operator *(Vector3 left, Vector3 right)
	{
		left.X *= right.X;
		left.Y *= right.Y;
		left.Z *= right.Z;
		return left;
	}

	public static Vector3 operator /(Vector3 vec, float divisor)
	{
		vec.X /= divisor;
		vec.Y /= divisor;
		vec.Z /= divisor;
		return vec;
	}

	public static Vector3 operator /(Vector3 vec, Vector3 divisorv)
	{
		vec.X /= divisorv.X;
		vec.Y /= divisorv.Y;
		vec.Z /= divisorv.Z;
		return vec;
	}

	public static Vector3 operator %(Vector3 vec, float divisor)
	{
		vec.X %= divisor;
		vec.Y %= divisor;
		vec.Z %= divisor;
		return vec;
	}

	public static Vector3 operator %(Vector3 vec, Vector3 divisorv)
	{
		vec.X %= divisorv.X;
		vec.Y %= divisorv.Y;
		vec.Z %= divisorv.Z;
		return vec;
	}

	public static bool operator ==(Vector3 left, Vector3 right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Vector3 left, Vector3 right)
	{
		return !left.Equals(right);
	}

	public static bool operator <(Vector3 left, Vector3 right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z < right.Z;
			}
			return left.Y < right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >(Vector3 left, Vector3 right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z > right.Z;
			}
			return left.Y > right.Y;
		}
		return left.X > right.X;
	}

	public static bool operator <=(Vector3 left, Vector3 right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z <= right.Z;
			}
			return left.Y < right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >=(Vector3 left, Vector3 right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z >= right.Z;
			}
			return left.Y > right.Y;
		}
		return left.X > right.X;
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Vector3 other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Vector3 other)
	{
		if (X == other.X && Y == other.Y)
		{
			return Z == other.Z;
		}
		return false;
	}

	public readonly bool IsEqualApprox(Vector3 other)
	{
		if (Mathf.IsEqualApprox(X, other.X) && Mathf.IsEqualApprox(Y, other.Y))
		{
			return Mathf.IsEqualApprox(Z, other.Z);
		}
		return false;
	}

	public readonly bool IsZeroApprox()
	{
		if (Mathf.IsZeroApprox(X) && Mathf.IsZeroApprox(Y))
		{
			return Mathf.IsZeroApprox(Z);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(X, Y, Z);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)})";
	}

	internal readonly Vector3 GetAnyPerpendicular()
	{
		if (IsZeroApprox())
		{
			throw new ArgumentException("The Vector3 must not be zero.");
		}
		return Cross((Mathf.Abs(X) <= Mathf.Abs(Y) && Mathf.Abs(X) <= Mathf.Abs(Z)) ? new Vector3(1f, 0f, 0f) : new Vector3(0f, 1f, 0f)).Normalized();
	}
}
