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

    // Dynamic call-by-name into the engine. Only reached on TestMode-gated audio/vfx paths that
    // early-return before invoking it headless (e.g. NAudioManager.PlayOneShot/StopAllLoops), so an
    // inert default suffices — and it must merely *exist* for those methods to JIT once an inert
    // stand-in singleton makes them reachable.
    public Variant Call(StringName method, params Variant[] args) => default;
}

public class RefCounted : GodotObject { }

public class Resource : RefCounted { }

// sts2 ships custom ResourceFormatLoader subclasses, listed (with every other script type) in the
// generator's [assembly: AssemblyHasScripts(Type[])]. Instantiating that attribute via reflection
// (xUnit scans assembly attributes) resolves every listed type, so the base must exist or the whole
// reflection throws a TypeLoadException. Inert — these loaders never run headless.
public class ResourceFormatLoader : RefCounted { }

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

// Particle process material. Referenced by particle-VFX setup that some card/event effects touch
// when their visuals load (e.g. the event heal VFX, curse-card glow); those code paths are
// TestMode-gated and never run headless, but their method bodies still JIT, so the properties they
// assign must exist (inert here). Real hierarchy: ParticleProcessMaterial : Material.
public class ParticleProcessMaterial : Material
{
    public Vector3 EmissionBoxExtents { get; set; }
    public Vector3 Gravity { get; set; }
    public Vector3 Direction { get; set; }
    public Color Color { get; set; } = Colors.White;
    public float EmissionSphereRadius { get; set; }
    public float InitialVelocityMin { get; set; }
    public float InitialVelocityMax { get; set; }
    public float Spread { get; set; }
    public float ScaleMin { get; set; }
    public float ScaleMax { get; set; }
}

public class Node : GodotObject
{
    /// <summary>Node name. Real Godot types it as a <see cref="StringName"/>; inert here.</summary>
    public StringName Name { get; set; } = "";

    // Visual-only; returns an inert tween. Headless callers either don't run (TestMode/
    // skipVisuals gated) or harmlessly drive a no-op tween.
    public Tween CreateTween() => new();

    public SceneTree GetTree() => SceneTree.Shared;

    // Scene-tree mutation/query API. Headless we never build a node tree (UI nodes stay
    // null/uninstantiated), so these are inert: no children, no parent, nothing in a tree.
    // They are referenced by VFX/card-pile traversal helpers (GodotTreeExtensions, CardCmd,
    // CardPileCmd, …) whose runtime branches are null-guarded and skipped headless; the members
    // only need to exist so those method bodies JIT.
    public enum InternalMode : long { Disabled, Front, Back }

    public Godot.Collections.Array<Node> GetChildren(bool includeInternal = false) => new();

    public int GetChildCount(bool includeInternal = false) => 0;

    public int GetIndex(bool includeInternal = false) => 0;

    public void AddChild(Node node, bool forceReadableName = false, InternalMode @internal = InternalMode.Disabled) { }

    public void AddSibling(Node sibling, bool forceReadableName = false) { }

    public void RemoveChild(Node node) { }

    public void MoveChild(Node childNode, int toIndex) { }

    // Node lookup by path. Headless there is no node tree, so every lookup misses (null). These
    // are referenced by monster/relic/room VFX code (e.g. Flyconid/SnappingJaxfruit spore moves,
    // NTreasureRoom's gold particles) whose call sites are TestMode-gated or null-conditional, so
    // the members only need to exist so those method bodies JIT — they are never invoked headless.
    public T GetNode<T>(NodePath path) where T : class => null!;

    public T GetNodeOrNull<T>(NodePath path) where T : class => null!;

    public Node GetNode(NodePath path) => null!;

    public Node GetNodeOrNull(NodePath path) => null!;

    public bool HasNode(NodePath path) => false;

    public Node? GetParent() => null;

    public Node? GetParentOrNull() => null;

    public bool IsAncestorOf(Node node) => false;

    public bool IsInsideTree() => false;

    public bool IsNodeReady() => false;

    public void QueueFree() { }

    public double GetProcessDeltaTime() => 1.0 / 60.0;

    public double GetPhysicsProcessDeltaTime() => 1.0 / 60.0;
}

public class CanvasItem : Node
{
    public Color Modulate { get; set; } = Colors.White;
    public Color SelfModulate { get; set; } = Colors.White;
    public bool Visible { get; set; } = true;
    public int ZIndex { get; set; }

    // Godot exposes the Visible property as explicit SetVisible/IsVisible methods, and the game's
    // (real-GodotSharp-compiled) IL calls those directly — e.g. monster setup toggling sprite
    // visibility (Crusher.AfterAddedToRoom). Route them through the same backing as the property.
    public void SetVisible(bool visible) => Visible = visible;
    public bool IsVisible() => Visible;

    public void Show() => Visible = true;
    public void Hide() => Visible = false;
    public void QueueRedraw() { }
    public void MoveToFront() { }

    // Viewport geometry queried by some VFX/layout code (e.g. particle placement when a card's
    // visuals load). Headless there is no viewport, so an empty rect is the inert answer.
    public Rect2 GetViewportRect() => default;
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
    public float RotationDegrees { get; set; }
    public float GlobalRotation { get; set; }
    public float GlobalRotationDegrees { get; set; }
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
public class Sprite2D : Node2D
{
    // Some monsters swap their sprite mid-fight (e.g. DecimillipedeSegment.ChangePhobiaModeTexture);
    // the texture swap is purely visual, so storing it inertly is enough to let the move run.
    public Texture2D? Texture { get; set; }
}
public class Marker2D : Node2D { }
// Particle VFX node. Referenced as the base of card-glow/treasure/boss VFX nodes and loaded by
// some monster moves (e.g. the CeremonialBeast act-1 boss) and card visuals (e.g. event-granted
// curses); headless it is only touched on visual paths, so the properties are inert. Real
// hierarchy: GpuParticles2D : Node2D.
public class GpuParticles2D : Node2D
{
    public Material? ProcessMaterial { get; set; }
    public Texture2D? Texture { get; set; }
    public bool Emitting { get; set; }
    public int Amount { get; set; }
    public float AmountRatio { get; set; }
    public double Lifetime { get; set; }
    public bool OneShot { get; set; }
    public double SpeedScale { get; set; } = 1.0;
}
public class AnimationPlayer : Node { }
public class CanvasLayer : Node { }
