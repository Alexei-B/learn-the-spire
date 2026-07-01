using System;
using System.Diagnostics.CodeAnalysis;

// The Godot C# source generator decorates a game assembly (and its Node subclasses) with these
// attributes: sts2.dll carries [assembly: AssemblyHasScripts(...)] plus per-class [ScriptPath]/
// [GodotClassName]. The shim never runs the generator, but anything that *reflects* over those
// custom attributes (e.g. xUnit's GetCustomAttributes<T>(assembly) scanning sts2.dll) must be able
// to load the attribute types or it throws a TypeLoadException — a catastrophic, non-test failure
// that fails the run even when every test passes. So define them inertly, copied verbatim from
// refsrc/GodotSharp. They are pure metadata (no native calls); reflection over them now resolves.
namespace Godot;

[AttributeUsage(AttributeTargets.Assembly)]
public sealed class AssemblyHasScriptsAttribute : Attribute
{
    [MemberNotNullWhen(false, "ScriptTypes")]
    public bool RequiresLookup
    {
        [MemberNotNullWhen(false, "ScriptTypes")]
        get;
    }

    public Type[]? ScriptTypes { get; }

    public AssemblyHasScriptsAttribute()
    {
        RequiresLookup = true;
        ScriptTypes = null;
    }

    public AssemblyHasScriptsAttribute(Type[] scriptTypes)
    {
        RequiresLookup = false;
        ScriptTypes = scriptTypes;
    }
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
public sealed class ScriptPathAttribute : Attribute
{
    public string Path { get; }

    public ScriptPathAttribute(string path)
    {
        Path = path;
    }
}

[AttributeUsage(AttributeTargets.Class)]
public class GodotClassNameAttribute : Attribute
{
    public string Name { get; }

    public GodotClassNameAttribute(string name)
    {
        Name = name;
    }
}
