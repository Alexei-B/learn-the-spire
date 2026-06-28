using System;

namespace Godot;

// Minimal Godot object/node hierarchy. The game's *logic* types are plain C# and
// never derive from these; they only appear here as referenced field/parameter
// types (e.g. a synchronizer holding a Control-typed VFX field) and as the base of
// UI nodes that we never instantiate headless (their N*.Instance accessors stay
// null). If we ever need to load a real Node subclass, this is where the Godot
// source-generator contract (nested MethodName/PropertyName/SignalName + virtual
// marshalling methods) would be added.

/// <summary>Headless base of the Godot object hierarchy.</summary>
public class GodotObject : IDisposable
{
    public GodotObject() { }

    public virtual void Dispose() => GC.SuppressFinalize(this);

    // Referenced by NonInteractiveMode-gated frame-wait branches; never executed headless.
    public SignalAwaiter ToSignal(GodotObject source, StringName signal) =>
        throw new InvalidOperationException("ToSignal is not available in the headless shim (no frame loop).");

    // Validity checks on engine objects. Headless we never hold disposed native handles, so a
    // non-null managed reference is always "valid" (and null is not). Used widely by logic that
    // guards against freed UI nodes (e.g. event-option hover tips).
    public static bool IsInstanceValid(GodotObject? instance) => instance is not null;
}

public class RefCounted : GodotObject { }

public class Resource : RefCounted { }

// Animation/scene types referenced by visual code paths that are skipped headless
// (skipVisuals / null-guarded UI). They only need to exist as types. Tween (with a
// real no-op API) lives in Tween.cs.
public class PackedScene : Resource
{
    public enum GenEditState : long { Disabled = 0, Instance = 1, Main = 2, MainInherited = 3 }

    // Headless there is no scene data to instantiate; return null. Callers add the
    // result to UI containers via null-guarded helpers, so null is harmless.
    public T? Instantiate<T>(GenEditState editState = GenEditState.Disabled) where T : Node => null;

    public Node? Instantiate(GenEditState editState = GenEditState.Disabled) => null;

    public bool CanInstantiate() => false;
}

public class SceneTreeTimer : RefCounted { }

public class Texture : Resource { }

public class Texture2D : Texture { }

public class CompressedTexture2D : Texture2D { }

public class AtlasTexture : Texture2D { }

public class ImageTexture : Texture2D { }

// Material/shader resources. Referenced as property/field types on card and relic models
// (e.g. for shader-driven visuals); headless they are only touched on skipVisuals paths, so
// the types just need to exist. The real hierarchy is Material : Resource, the concrete
// materials deriving from it.
public class Material : Resource { }

public class ShaderMaterial : Material { }

public class Shader : Resource { }

public class Node : GodotObject
{
    /// <summary>Node name. Real Godot types it as a <see cref="StringName"/>; inert here.</summary>
    public StringName Name { get; set; } = "";

    // Visual-only; returns an inert tween. Headless callers either don't run (TestMode/
    // skipVisuals gated) or harmlessly drive a no-op tween.
    public Tween CreateTween() => new();

    public SceneTree GetTree() => SceneTree.Shared;

    public double GetProcessDeltaTime() => 1.0 / 60.0;

    public double GetPhysicsProcessDeltaTime() => 1.0 / 60.0;
}

public class CanvasItem : Node
{
    public Color Modulate { get; set; } = Colors.White;
    public Color SelfModulate { get; set; } = Colors.White;
    public bool Visible { get; set; } = true;
    public int ZIndex { get; set; }

    public void Show() => Visible = true;
    public void Hide() => Visible = false;
    public void QueueRedraw() { }
}

public class Control : CanvasItem
{
    public Vector2 Position { get; set; }
    public Vector2 GlobalPosition { get; set; }
    public Vector2 Size { get; set; }
    public Vector2 CustomMinimumSize { get; set; }
    public Vector2 Scale { get; set; } = Vector2.One;
    public Vector2 PivotOffset { get; set; }
    public float Rotation { get; set; }
}

public class Node2D : CanvasItem
{
    public Vector2 Position { get; set; }
    public Vector2 GlobalPosition { get; set; }
    public Vector2 Scale { get; set; } = Vector2.One;
    public Vector2 GlobalScale { get; set; } = Vector2.One;
    public float Rotation { get; set; }
    public float GlobalRotation { get; set; }
    public float Skew { get; set; }
    public int ZIndexNode2D { get; set; }
}

public class Viewport : Node { }

public class Window : Viewport { }

public class MainLoop : GodotObject { }

public class SceneTree : MainLoop
{
    /// <summary>Process-wide stand-in scene tree (there is only ever one, headless).</summary>
    public static readonly SceneTree Shared = new();

    private readonly Window _root = new();

    public Window Root => _root;

    // Referenced only by the NonInteractiveMode-gated delay path; never created headless.
    public SceneTreeTimer CreateTimer(double timeSec, bool processAlways = true, bool processInPhysics = false, bool ignoreTimeScale = false) => new();

    public class SignalName
    {
        public static readonly StringName ProcessFrame = "process_frame";
        public static readonly StringName PhysicsFrame = "physics_frame";
    }
}

// Common UI node types. The game references these as field/local/parameter types in
// room/combat flow, but headless they are only ever touched through null-guarded
// N*.Instance accessors, so empty subclasses that merely exist are sufficient.
public class Container : Control { }
public class ColorRect : Control { }
public class Label : Control { }
public class RichTextLabel : Control { }
public class TextureRect : Control { }
public class Panel : Control { }
public class PanelContainer : Container { }
public class MarginContainer : Container { }
public class BoxContainer : Container { }
public class VBoxContainer : BoxContainer { }
public class HBoxContainer : BoxContainer { }
public class CenterContainer : Container { }
public class GridContainer : Container { }
public class FlowContainer : Container { }
public class HFlowContainer : FlowContainer { }
public class VFlowContainer : FlowContainer { }
public class Button : Control { }
public class TextureButton : Control { }
public class Sprite2D : Node2D { }
public class Marker2D : Node2D { }
public class AnimationPlayer : Node { }
public class CanvasLayer : Node { }
