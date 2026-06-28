namespace Godot;

/// <summary>
/// No-op replacement for Godot's <c>Tween</c> and its tweeners. Every animation in
/// the game runs through these; headless they do nothing. The chainable API is fully
/// reproduced so the (mostly TestMode/skipVisuals-gated) call sites JIT and run
/// without effect. Enum values match GodotSharp.
/// </summary>
public class Tween : RefCounted
{
    public enum TweenProcessMode : long { Physics = 0, Idle = 1 }

    public enum TweenPauseMode : long { Bound = 0, Stop = 1, Process = 2 }

    public enum TransitionType : long
    {
        Linear = 0, Sine = 1, Quint = 2, Quart = 3, Quad = 4, Expo = 5,
        Elastic = 6, Cubic = 7, Circ = 8, Bounce = 9, Back = 10, Spring = 11,
    }

    public enum EaseType : long { In = 0, Out = 1, InOut = 2, OutIn = 3 }

    public PropertyTweener TweenProperty(GodotObject @object, NodePath property, Variant finalVal, double duration) => new();

    public IntervalTweener TweenInterval(double time) => new();

    public CallbackTweener TweenCallback(Callable callback) => new();

    public MethodTweener TweenMethod(Callable method, Variant from, Variant to, double duration) => new();

    public bool CustomStep(double delta) => false;

    public void Stop() { }

    public void Pause() { }

    public void Play() { }

    public void Kill() { }

    public bool IsRunning() => false;

    public bool IsValid() => false;

    public Tween BindNode(Node node) => this;

    public Tween SetParallel(bool parallel = true) => this;

    public Tween SetLoops(int loops = 0) => this;

    public Tween SetTrans(TransitionType trans) => this;

    public Tween SetEase(EaseType ease) => this;

    public Tween SetProcessMode(TweenProcessMode mode) => this;

    public Tween SetPauseMode(TweenPauseMode mode) => this;

    public Tween Parallel() => this;

    public Tween Chain() => this;
}

public class Tweener : RefCounted { }

public sealed class PropertyTweener : Tweener
{
    public PropertyTweener From(Variant value) => this;
    public PropertyTweener FromCurrent() => this;
    public PropertyTweener AsRelative() => this;
    public PropertyTweener SetTrans(Tween.TransitionType trans) => this;
    public PropertyTweener SetEase(Tween.EaseType ease) => this;
    public PropertyTweener SetDelay(double delay) => this;
}

public sealed class IntervalTweener : Tweener { }

public sealed class CallbackTweener : Tweener
{
    public CallbackTweener SetDelay(double delay) => this;
}

public sealed class MethodTweener : Tweener
{
    public MethodTweener SetDelay(double delay) => this;
    public MethodTweener SetEase(Tween.EaseType ease) => this;
    public MethodTweener SetTrans(Tween.TransitionType trans) => this;
}
