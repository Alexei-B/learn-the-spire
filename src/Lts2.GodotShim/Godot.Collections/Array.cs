using System.Collections;
using System.Collections.Generic;

namespace Godot.Collections;

/// <summary>
/// Headless replacement for Godot's variant-backed <c>Array&lt;T&gt;</c>, implemented
/// as a plain managed list. See <see cref="Dictionary{TKey,TValue}"/> for rationale.
/// </summary>
public sealed class Array<T> : IList<T>, IReadOnlyList<T>
{
    private readonly List<T> _inner;

    public Array() => _inner = new List<T>();
    public Array(IEnumerable<T> collection) => _inner = new List<T>(collection);

    public T this[int index] { get => _inner[index]; set => _inner[index] = value; }
    public int Count => _inner.Count;
    public bool IsReadOnly => false;

    public void Add(T item) => _inner.Add(item);
    public void Clear() => _inner.Clear();
    public bool Contains(T item) => _inner.Contains(item);
    public void CopyTo(T[] array, int arrayIndex) => _inner.CopyTo(array, arrayIndex);
    public int IndexOf(T item) => _inner.IndexOf(item);
    public void Insert(int index, T item) => _inner.Insert(index, item);
    public bool Remove(T item) => _inner.Remove(item);
    public void RemoveAt(int index) => _inner.RemoveAt(index);

    public IEnumerator<T> GetEnumerator() => _inner.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => _inner.GetEnumerator();
}
