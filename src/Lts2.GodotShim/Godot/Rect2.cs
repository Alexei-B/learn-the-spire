using System;

namespace Godot;

[Serializable]
public struct Rect2 : IEquatable<Rect2>
{
    private Vector2 _position;
    private Vector2 _size;

    public Vector2 Position
    {
        readonly get => _position;
        set => _position = value;
    }

    public Vector2 Size
    {
        readonly get => _size;
        set => _size = value;
    }

    public Vector2 End
    {
        readonly get => _position + _size;
        set => _size = value - _position;
    }

    public readonly float Area => _size.X * _size.Y;

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

    public readonly Rect2 Abs()
    {
        Vector2 position = End.Min(_position);
        return new Rect2(position, _size.Abs());
    }

    public readonly Rect2 Intersection(Rect2 b)
    {
        if (!Intersects(b))
        {
            return default;
        }

        Vector2 position = b._position.Max(_position);
        Vector2 end = (b._position + b._size).Min(_position + _size);
        return new Rect2(position, end - position);
    }

    public bool IsFinite() => _position.IsFinite() && _size.IsFinite();

    public readonly bool Encloses(Rect2 b)
    {
        return b._position.X >= _position.X
            && b._position.Y >= _position.Y
            && b._position.X + b._size.X <= _position.X + _size.X
            && b._position.Y + b._size.Y <= _position.Y + _size.Y;
    }

    public readonly Rect2 Expand(Vector2 to)
    {
        Vector2 begin = _position;
        Vector2 end = _position + _size;

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

        return new Rect2(begin, end - begin);
    }

    public readonly Vector2 GetCenter() => _position + _size * 0.5f;

    public readonly Vector2 GetSupport(Vector2 direction)
    {
        Vector2 support = _position;
        if (direction.X > 0f)
        {
            support.X += _size.X;
        }

        if (direction.Y > 0f)
        {
            support.Y += _size.Y;
        }

        return support;
    }

    public readonly Rect2 Grow(float by)
    {
        return new Rect2(
            _position.X - by,
            _position.Y - by,
            _size.X + by * 2f,
            _size.Y + by * 2f);
    }

    public readonly Rect2 GrowIndividual(float left, float top, float right, float bottom)
    {
        return new Rect2(
            _position.X - left,
            _position.Y - top,
            _size.X + left + right,
            _size.Y + top + bottom);
    }

    public readonly Rect2 GrowSide(Side side, float by)
    {
        return GrowIndividual(
            side == Side.Left ? by : 0f,
            side == Side.Top ? by : 0f,
            side == Side.Right ? by : 0f,
            side == Side.Bottom ? by : 0f);
    }

    public readonly bool HasArea() => _size.X > 0f && _size.Y > 0f;

    public readonly bool HasPoint(Vector2 point)
    {
        return point.X >= _position.X
            && point.Y >= _position.Y
            && point.X < _position.X + _size.X
            && point.Y < _position.Y + _size.Y;
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
        Vector2 begin = _position.Min(b._position);
        Vector2 end = (_position + _size).Max(b._position + b._size);
        return new Rect2(begin, end - begin);
    }

    public readonly bool Equals(Rect2 other)
    {
        return _position.Equals(other._position) && _size.Equals(other._size);
    }

    public readonly bool IsEqualApprox(Rect2 other)
    {
        return _position.IsEqualApprox(other._position) && _size.IsEqualApprox(other._size);
    }

    public override readonly bool Equals(object? obj) => obj is Rect2 other && Equals(other);

    public override readonly int GetHashCode() => HashCode.Combine(_position, _size);

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"{_position.ToString(format)}, {_size.ToString(format)}";
    }

    public static bool operator ==(Rect2 left, Rect2 right) => left.Equals(right);

    public static bool operator !=(Rect2 left, Rect2 right) => !left.Equals(right);
}
