using System;
using System.Linq;

namespace Godot;

/// <summary>
/// Headless replacement for Godot's <c>Callable</c>. The real type wraps a delegate
/// plus a native trampoline so the engine can invoke managed methods. Headless we
/// own invocation, so we just store the delegate and call it directly. "Deferred"
/// calls run immediately — there is no frame loop to defer to, and the game's
/// no-delay mode expects synchronous resolution anyway.
/// </summary>
public readonly struct Callable
{
    private readonly Delegate? _delegate;

    private Callable(Delegate? @delegate) => _delegate = @delegate;

    public Delegate? Delegate => _delegate;

    public static Callable From(Action action) => new(action);

    public static Callable From<T0>(Action<T0> action) => new(action);

    public static Callable From<T0, T1>(Action<T0, T1> action) => new(action);

    public static Callable From<TResult>(Func<TResult> func) => new(func);

    public Variant Call(params Variant[] args)
    {
        object? result = Invoke(args);
        return result is null ? default : Variant.From(result);
    }

    public void CallDeferred(params Variant[] args) => Invoke(args);

    private object? Invoke(Variant[] args)
    {
        if (_delegate is null)
        {
            return null;
        }
        if (args is null || args.Length == 0)
        {
            return _delegate.DynamicInvoke();
        }
        return _delegate.DynamicInvoke(args.Select(a => a.Obj).ToArray());
    }
}
