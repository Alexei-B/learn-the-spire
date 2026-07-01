using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Godot;

[Serializable]
public struct Vector2I : IEquatable<Vector2I>
{
	public enum Axis
	{
		X,
		Y
	}

	public int X;

	public int Y;

	private static readonly Vector2I _minValue = new Vector2I(int.MinValue, int.MinValue);

	private static readonly Vector2I _maxValue = new Vector2I(int.MaxValue, int.MaxValue);

	private static readonly Vector2I _zero = new Vector2I(0, 0);

	private static readonly Vector2I _one = new Vector2I(1, 1);

	private static readonly Vector2I _up = new Vector2I(0, -1);

	private static readonly Vector2I _down = new Vector2I(0, 1);

	private static readonly Vector2I _right = new Vector2I(1, 0);

	private static readonly Vector2I _left = new Vector2I(-1, 0);

	public int this[int index]
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

	public readonly Vector2I Abs()
	{
		return new Vector2I(Mathf.Abs(X), Mathf.Abs(Y));
	}

	public readonly float Aspect()
	{
		return (float)X / (float)Y;
	}

	public readonly Vector2I Clamp(Vector2I min, Vector2I max)
	{
		return new Vector2I(Mathf.Clamp(X, min.X, max.X), Mathf.Clamp(Y, min.Y, max.Y));
	}

	public readonly Vector2I Clamp(int min, int max)
	{
		return new Vector2I(Mathf.Clamp(X, min, max), Mathf.Clamp(Y, min, max));
	}

	public readonly int DistanceSquaredTo(Vector2I to)
	{
		return (to - this).LengthSquared();
	}

	public readonly float DistanceTo(Vector2I to)
	{
		return (to - this).Length();
	}

	public readonly float Length()
	{
		int num = X * X;
		int num2 = Y * Y;
		return Mathf.Sqrt(num + num2);
	}

	public readonly int LengthSquared()
	{
		int num = X * X;
		int num2 = Y * Y;
		return num + num2;
	}

	public readonly Vector2I Max(Vector2I with)
	{
		return new Vector2I(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y));
	}

	public readonly Vector2I Max(int with)
	{
		return new Vector2I(Mathf.Max(X, with), Mathf.Max(Y, with));
	}

	public readonly Vector2I Min(Vector2I with)
	{
		return new Vector2I(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y));
	}

	public readonly Vector2I Min(int with)
	{
		return new Vector2I(Mathf.Min(X, with), Mathf.Min(Y, with));
	}

	public readonly Axis MaxAxisIndex()
	{
		if (X >= Y)
		{
			return Axis.X;
		}
		return Axis.Y;
	}

	public readonly Axis MinAxisIndex()
	{
		if (X >= Y)
		{
			return Axis.Y;
		}
		return Axis.X;
	}

	public readonly Vector2I Sign()
	{
		Vector2I result = this;
		result.X = Mathf.Sign(result.X);
		result.Y = Mathf.Sign(result.Y);
		return result;
	}

	public readonly Vector2I Snapped(Vector2I step)
	{
		return new Vector2I((int)Mathf.Snapped((double)X, (double)step.X), (int)Mathf.Snapped((double)Y, (double)step.Y));
	}

	public readonly Vector2I Snapped(int step)
	{
		return new Vector2I((int)Mathf.Snapped((double)X, (double)step), (int)Mathf.Snapped((double)Y, (double)step));
	}

	public Vector2I(int x, int y)
	{
		X = x;
		Y = y;
	}

	public static Vector2I operator +(Vector2I left, Vector2I right)
	{
		left.X += right.X;
		left.Y += right.Y;
		return left;
	}

	public static Vector2I operator -(Vector2I left, Vector2I right)
	{
		left.X -= right.X;
		left.Y -= right.Y;
		return left;
	}

	public static Vector2I operator -(Vector2I vec)
	{
		vec.X = -vec.X;
		vec.Y = -vec.Y;
		return vec;
	}

	public static Vector2I operator *(Vector2I vec, int scale)
	{
		vec.X *= scale;
		vec.Y *= scale;
		return vec;
	}

	public static Vector2I operator *(int scale, Vector2I vec)
	{
		vec.X *= scale;
		vec.Y *= scale;
		return vec;
	}

	public static Vector2I operator *(Vector2I left, Vector2I right)
	{
		left.X *= right.X;
		left.Y *= right.Y;
		return left;
	}

	public static Vector2I operator /(Vector2I vec, int divisor)
	{
		vec.X /= divisor;
		vec.Y /= divisor;
		return vec;
	}

	public static Vector2I operator /(Vector2I vec, Vector2I divisorv)
	{
		vec.X /= divisorv.X;
		vec.Y /= divisorv.Y;
		return vec;
	}

	public static Vector2I operator %(Vector2I vec, int divisor)
	{
		vec.X %= divisor;
		vec.Y %= divisor;
		return vec;
	}

	public static Vector2I operator %(Vector2I vec, Vector2I divisorv)
	{
		vec.X %= divisorv.X;
		vec.Y %= divisorv.Y;
		return vec;
	}

	public static bool operator ==(Vector2I left, Vector2I right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Vector2I left, Vector2I right)
	{
		return !left.Equals(right);
	}

	public static bool operator <(Vector2I left, Vector2I right)
	{
		if (left.X == right.X)
		{
			return left.Y < right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >(Vector2I left, Vector2I right)
	{
		if (left.X == right.X)
		{
			return left.Y > right.Y;
		}
		return left.X > right.X;
	}

	public static bool operator <=(Vector2I left, Vector2I right)
	{
		if (left.X == right.X)
		{
			return left.Y <= right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >=(Vector2I left, Vector2I right)
	{
		if (left.X == right.X)
		{
			return left.Y >= right.Y;
		}
		return left.X > right.X;
	}

	public static implicit operator Vector2(Vector2I value)
	{
		return new Vector2(value.X, value.Y);
	}

	public static explicit operator Vector2I(Vector2 value)
	{
		return new Vector2I((int)value.X, (int)value.Y);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Vector2I other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Vector2I other)
	{
		if (X == other.X)
		{
			return Y == other.Y;
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
