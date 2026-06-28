using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace Godot.Collections;

/// <summary>
/// Headless replacement for Godot's variant-backed <c>Dictionary&lt;TKey,TValue&gt;</c>.
/// The real type marshals through a native godot_dictionary; here it is a plain
/// managed dictionary. The game uses Godot.Collections only in infrastructure code
/// (never in core game logic), so faithful variant semantics are unnecessary.
/// </summary>
public class Dictionary<TKey, TValue> : IDictionary<TKey, TValue>, IReadOnlyDictionary<TKey, TValue> where TKey : notnull
{
    private readonly System.Collections.Generic.Dictionary<TKey, TValue> _inner;

    public Dictionary() => _inner = new System.Collections.Generic.Dictionary<TKey, TValue>();

    public Dictionary(IDictionary<TKey, TValue> dictionary) =>
        _inner = new System.Collections.Generic.Dictionary<TKey, TValue>(dictionary);

    public TValue this[TKey key] { get => _inner[key]; set => _inner[key] = value; }

    public ICollection<TKey> Keys => _inner.Keys;
    public ICollection<TValue> Values => _inner.Values;
    IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => _inner.Keys;
    IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values => _inner.Values;

    public int Count => _inner.Count;
    public bool IsReadOnly => false;

    public void Add(TKey key, TValue value) => _inner.Add(key, value);
    public void Add(KeyValuePair<TKey, TValue> item) => _inner.Add(item.Key, item.Value);
    public void Clear() => _inner.Clear();
    public bool Contains(KeyValuePair<TKey, TValue> item) => ((ICollection<KeyValuePair<TKey, TValue>>)_inner).Contains(item);
    public bool ContainsKey(TKey key) => _inner.ContainsKey(key);
    public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex) => ((ICollection<KeyValuePair<TKey, TValue>>)_inner).CopyTo(array, arrayIndex);
    public bool Remove(TKey key) => _inner.Remove(key);
    public bool Remove(KeyValuePair<TKey, TValue> item) => ((ICollection<KeyValuePair<TKey, TValue>>)_inner).Remove(item);
    public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value) => _inner.TryGetValue(key, out value);

    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() => _inner.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => _inner.GetEnumerator();
}
