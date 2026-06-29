namespace Godot;

// Godot node/resource types that sts2 subclasses as *scripts* but the harness never instantiates
// (UI input widgets, particle/line VFX, path animation, rich-text effects). They appear in the
// generator's [assembly: AssemblyHasScripts(Type[])]; reflecting that attribute (xUnit scans
// assembly custom attributes during a test run) resolves every listed script type, so each base
// must exist or the whole reflection throws a TypeLoadException — a catastrophic, non-test failure
// that fails `dotnet test` with exit code 1 even when every test passes. Inert facades that merely
// exist are sufficient (the script subclasses are never loaded onto a live scene headless).

public class LineEdit : Control { }
public class TextEdit : Control { }
public class Range : Control { }
public class AspectRatioContainer : Container { }

public class Line2D : Node2D { }
public class BackBufferCopy : Node2D { }
public class CpuParticles2D : Node2D { }
public class PathFollow2D : Node2D { }

public class RichTextEffect : RefCounted { }

// Input keycodes; referenced only as an enum type by script code paths never executed headless.
public enum Key : long
{
    None = 0,
}
