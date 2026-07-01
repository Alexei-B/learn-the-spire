using System;
using System.Diagnostics.CodeAnalysis;

namespace Godot;

[Serializable]
public struct Rect2I : IEquatable<Rect2I>
{
	private Vector2I _position;

	private Vector2I _size;

	public Vector2I Position
	{
		readonly get
		{
			return _position;
		}
		set
		{
			_position = value;
		}
	}

	public Vector2I Size
	{
		readonly get
		{
			return _size;
		}
		set
		{
			_size = value;
		}
	}

	public Vector2I End
	{
		readonly get
		{
			return _position + _size;
		}
		set
		{
			_size = value - _position;
		}
	}

	public readonly int Area => _size.X * _size.Y;

	public readonly Rect2I Abs()
	{
		return new Rect2I(End.Min(_position), _size.Abs());
	}

	public readonly Rect2I Intersection(Rect2I b)
	{
		Rect2I rect2I = b;
		if (!Intersects(rect2I))
		{
			return default(Rect2I);
		}
		rect2I._position = b._position.Max(_position);
		Vector2I vector2I = b._position + b._size;
		Vector2I with = _position + _size;
		rect2I._size = vector2I.Min(with) - rect2I._position;
		return rect2I;
	}

	public readonly bool Encloses(Rect2I b)
	{
		if (b._position.X >= _position.X && b._position.Y >= _position.Y && b._position.X + b._size.X <= _position.X + _size.X)
		{
			return b._position.Y + b._size.Y <= _position.Y + _size.Y;
		}
		return false;
	}

	public readonly Rect2I Expand(Vector2I to)
	{
		Rect2I result = this;
		Vector2I position = result._position;
		Vector2I vector2I = result._position + result._size;
		if (to.X < position.X)
		{
			position.X = to.X;
		}
		if (to.Y < position.Y)
		{
			position.Y = to.Y;
		}
		if (to.X > vector2I.X)
		{
			vector2I.X = to.X;
		}
		if (to.Y > vector2I.Y)
		{
			vector2I.Y = to.Y;
		}
		result._position = position;
		result._size = vector2I - position;
		return result;
	}

	public readonly Vector2I GetCenter()
	{
		return _position + _size / 2;
	}

	public readonly Rect2I Grow(int by)
	{
		Rect2I result = this;
		result._position.X -= by;
		result._position.Y -= by;
		result._size.X += by * 2;
		result._size.Y += by * 2;
		return result;
	}

	public readonly Rect2I GrowIndividual(int left, int top, int right, int bottom)
	{
		Rect2I result = this;
		result._position.X -= left;
		result._position.Y -= top;
		result._size.X += left + right;
		result._size.Y += top + bottom;
		return result;
	}

	public readonly Rect2I GrowSide(Side side, int by)
	{
		Rect2I rect2I = this;
		return rect2I.GrowIndividual((side == Side.Left) ? by : 0, (Side.Top == side) ? by : 0, (Side.Right == side) ? by : 0, (Side.Bottom == side) ? by : 0);
	}

	public readonly bool HasArea()
	{
		if (_size.X > 0)
		{
			return _size.Y > 0;
		}
		return false;
	}

	public readonly bool HasPoint(Vector2I point)
	{
		if (point.X < _position.X)
		{
			return false;
		}
		if (point.Y < _position.Y)
		{
			return false;
		}
		if (point.X >= _position.X + _size.X)
		{
			return false;
		}
		if (point.Y >= _position.Y + _size.Y)
		{
			return false;
		}
		return true;
	}

	public readonly bool Intersects(Rect2I b)
	{
		if (_position.X >= b._position.X + b._size.X)
		{
			return false;
		}
		if (_position.X + _size.X <= b._position.X)
		{
			return false;
		}
		if (_position.Y >= b._position.Y + b._size.Y)
		{
			return false;
		}
		if (_position.Y + _size.Y <= b._position.Y)
		{
			return false;
		}
		return true;
	}

	public readonly Rect2I Merge(Rect2I b)
	{
		Rect2I result = default(Rect2I);
		result._position = b._position.Min(_position);
		result._size = (b._position + b._size).Max(_position + _size);
		result._size -= result._position;
		return result;
	}

	public Rect2I(Vector2I position, Vector2I size)
	{
		_position = position;
		_size = size;
	}

	public Rect2I(Vector2I position, int width, int height)
	{
		_position = position;
		_size = new Vector2I(width, height);
	}

	public Rect2I(int x, int y, Vector2I size)
	{
		_position = new Vector2I(x, y);
		_size = size;
	}

	public Rect2I(int x, int y, int width, int height)
	{
		_position = new Vector2I(x, y);
		_size = new Vector2I(width, height);
	}

	public static bool operator ==(Rect2I left, Rect2I right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Rect2I left, Rect2I right)
	{
		return !left.Equals(right);
	}

	public static implicit operator Rect2(Rect2I value)
	{
		return new Rect2(value._position, value._size);
	}

	public static explicit operator Rect2I(Rect2 value)
	{
		return new Rect2I((Vector2I)value.Position, (Vector2I)value.Size);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Rect2I other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Rect2I other)
	{
		if (_position.Equals(other._position))
		{
			return _size.Equals(other._size);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(_position, _size);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return _position.ToString(format) + ", " + _size.ToString(format);
	}
}
