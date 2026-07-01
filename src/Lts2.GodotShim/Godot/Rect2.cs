using System;
using System.Diagnostics.CodeAnalysis;

namespace Godot;

[Serializable]
public struct Rect2 : IEquatable<Rect2>
{
	private Vector2 _position;

	private Vector2 _size;

	public Vector2 Position
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

	public Vector2 Size
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

	public Vector2 End
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

	public readonly float Area => _size.X * _size.Y;

	public readonly Rect2 Abs()
	{
		return new Rect2(End.Min(_position), _size.Abs());
	}

	public readonly Rect2 Intersection(Rect2 b)
	{
		Rect2 rect = b;
		if (!Intersects(rect))
		{
			return default(Rect2);
		}
		rect._position = b._position.Max(_position);
		Vector2 vector = b._position + b._size;
		Vector2 with = _position + _size;
		rect._size = vector.Min(with) - rect._position;
		return rect;
	}

	public bool IsFinite()
	{
		if (_position.IsFinite())
		{
			return _size.IsFinite();
		}
		return false;
	}

	public readonly bool Encloses(Rect2 b)
	{
		if (b._position.X >= _position.X && b._position.Y >= _position.Y && b._position.X + b._size.X <= _position.X + _size.X)
		{
			return b._position.Y + b._size.Y <= _position.Y + _size.Y;
		}
		return false;
	}

	public readonly Rect2 Expand(Vector2 to)
	{
		Rect2 result = this;
		Vector2 position = result._position;
		Vector2 vector = result._position + result._size;
		if (to.X < position.X)
		{
			position.X = to.X;
		}
		if (to.Y < position.Y)
		{
			position.Y = to.Y;
		}
		if (to.X > vector.X)
		{
			vector.X = to.X;
		}
		if (to.Y > vector.Y)
		{
			vector.Y = to.Y;
		}
		result._position = position;
		result._size = vector - position;
		return result;
	}

	public readonly Vector2 GetCenter()
	{
		return _position + _size * 0.5f;
	}

	public readonly Vector2 GetSupport(Vector2 direction)
	{
		Vector2 position = _position;
		if (direction.X > 0f)
		{
			position.X += _size.X;
		}
		if (direction.Y > 0f)
		{
			position.Y += _size.Y;
		}
		return position;
	}

	public readonly Rect2 Grow(float by)
	{
		Rect2 result = this;
		result._position.X -= by;
		result._position.Y -= by;
		result._size.X += by * 2f;
		result._size.Y += by * 2f;
		return result;
	}

	public readonly Rect2 GrowIndividual(float left, float top, float right, float bottom)
	{
		Rect2 result = this;
		result._position.X -= left;
		result._position.Y -= top;
		result._size.X += left + right;
		result._size.Y += top + bottom;
		return result;
	}

	public readonly Rect2 GrowSide(Side side, float by)
	{
		Rect2 rect = this;
		return rect.GrowIndividual((side == Side.Left) ? by : 0f, (Side.Top == side) ? by : 0f, (Side.Right == side) ? by : 0f, (Side.Bottom == side) ? by : 0f);
	}

	public readonly bool HasArea()
	{
		if (_size.X > 0f)
		{
			return _size.Y > 0f;
		}
		return false;
	}

	public readonly bool HasPoint(Vector2 point)
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

	public readonly bool Intersects(Rect2 b, bool includeBorders = false)
	{
		if (includeBorders)
		{
			if (_position.X > b._position.X + b._size.X)
			{
				return false;
			}
			if (_position.X + _size.X < b._position.X)
			{
				return false;
			}
			if (_position.Y > b._position.Y + b._size.Y)
			{
				return false;
			}
			if (_position.Y + _size.Y < b._position.Y)
			{
				return false;
			}
		}
		else
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
		}
		return true;
	}

	public readonly Rect2 Merge(Rect2 b)
	{
		Rect2 result = default(Rect2);
		result._position = b._position.Min(_position);
		result._size = (b._position + b._size).Max(_position + _size);
		result._size -= result._position;
		return result;
	}

	public Rect2(Vector2 position, Vector2 size)
	{
		_position = position;
		_size = size;
	}

	public Rect2(Vector2 position, float width, float height)
	{
		_position = position;
		_size = new Vector2(width, height);
	}

	public Rect2(float x, float y, Vector2 size)
	{
		_position = new Vector2(x, y);
		_size = size;
	}

	public Rect2(float x, float y, float width, float height)
	{
		_position = new Vector2(x, y);
		_size = new Vector2(width, height);
	}

	public static bool operator ==(Rect2 left, Rect2 right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Rect2 left, Rect2 right)
	{
		return !left.Equals(right);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Rect2 other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Rect2 other)
	{
		if (_position.Equals(other._position))
		{
			return _size.Equals(other._size);
		}
		return false;
	}

	public readonly bool IsEqualApprox(Rect2 other)
	{
		if (_position.IsEqualApprox(other._position))
		{
			return _size.IsEqualApprox(other.Size);
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
