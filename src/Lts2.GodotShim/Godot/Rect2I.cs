using System;

namespace Godot;

[Serializable]
public struct Rect2I : IEquatable<Rect2I>
{
    private Vector2I _position;
    private Vector2I _size;

    public Vector2I Position
    {
        readonly get => _position;
        set => _position = value;
    }

    public Vector2I Size
    {
        readonly get => _size;
        set => _size = value;
    }

    public Vector2I End
    {
        readonly get => _position + _size;
        set => _size = value - _position;
    }

    public readonly int Area => _size.X * _size.Y;

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

    public readonly Rect2I Abs()
    {
        Vector2I position = End.Min(_position);
        return new Rect2I(position, _size.Abs());
    }

    public readonly Rect2I Intersection(Rect2I b)
    {
        if (!Intersects(b))
        {
            return default;
        }

        Vector2I position = b._position.Max(_position);
        Vector2I end = (b._position + b._size).Min(_position + _size);
        return new Rect2I(position, end - position);
    }

    public readonly bool Encloses(Rect2I b)
    {
        return b._position.X >= _position.X
            && b._position.Y >= _position.Y
            && b._position.X + b._size.X <= _position.X + _size.X
            && b._position.Y + b._size.Y <= _position.Y + _size.Y;
    }

    public readonly Rect2I Expand(Vector2I to)
    {
        Vector2I begin = _position;
        Vector2I end = _position + _size;

        if (to.X < begin.X)
        {
            begin.X = to.X;
        }

        if (to.Y < begin.Y)
        {
            begin.Y = to.Y;
        }

        if (to.X > end.X)
        {
            end.X = to.X;
        }

        if (to.Y > end.Y)
        {
            end.Y = to.Y;
        }

        return new Rect2I(begin, end - begin);
    }

    public readonly Vector2I GetCenter() => _position + _size / 2;

    public readonly Rect2I Grow(int by)
    {
        return new Rect2I(
            _position.X - by,
            _position.Y - by,
            _size.X + by * 2,
            _size.Y + by * 2);
    }

    public readonly Rect2I GrowIndividual(int left, int top, int right, int bottom)
    {
        return new Rect2I(
            _position.X - left,
            _position.Y - top,
            _size.X + left + right,
            _size.Y + top + bottom);
    }

    public readonly Rect2I GrowSide(Side side, int by)
    {
        return GrowIndividual(
            side == Side.Left ? by : 0,
            side == Side.Top ? by : 0,
            side == Side.Right ? by : 0,
            side == Side.Bottom ? by : 0);
    }

    public readonly bool HasArea() => _size.X > 0 && _size.Y > 0;

    public readonly bool HasPoint(Vector2I point)
    {
        return point.X >= _position.X
            && point.Y >= _position.Y
            && point.X < _position.X + _size.X
            && point.Y < _position.Y + _size.Y;
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
        Vector2I begin = _position.Min(b._position);
        Vector2I end = (_position + _size).Max(b._position + b._size);
        return new Rect2I(begin, end - begin);
    }

    public readonly bool Equals(Rect2I other)
    {
        return _position.Equals(other._position) && _size.Equals(other._size);
    }

    public override readonly bool Equals(object? obj) => obj is Rect2I other && Equals(other);

    public override readonly int GetHashCode() => HashCode.Combine(_position, _size);

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"{_position.ToString(format)}, {_size.ToString(format)}";
    }

    public static bool operator ==(Rect2I left, Rect2I right) => left.Equals(right);

    public static bool operator !=(Rect2I left, Rect2I right) => !left.Equals(right);

    public static implicit operator Rect2(Rect2I value)
    {
        return new Rect2(value._position, value._size);
    }

    public static explicit operator Rect2I(Rect2 value)
    {
        return new Rect2I((Vector2I)value.Position, (Vector2I)value.Size);
    }
}
