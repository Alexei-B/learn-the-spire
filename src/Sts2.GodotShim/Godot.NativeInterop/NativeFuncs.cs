namespace Godot.NativeInterop;

/// <summary>
/// Headless replacement for Godot's native-interop entry points.
///
/// In the real GodotSharp these forward to a table of function pointers that the
/// native Godot engine fills in at startup. We have no engine, so every entry here
/// throws a clean, catchable <see cref="System.NotSupportedException"/> instead of
/// dereferencing a null function pointer (which would be an uncatchable
/// AccessViolationException). Pure-managed value-type math never reaches these;
/// only genuinely engine-backed operations do, and those are out of scope headless.
///
/// Methods are added here on demand as shimmed types reference them.
/// </summary>
internal static class NativeFuncs
{
    private static System.NotSupportedException NotAvailable(string name) =>
        new($"Godot native call '{name}' is not available in the headless shim.");

    internal static float godotsharp_color_get_ok_hsl_h(in Color self) =>
        throw NotAvailable(nameof(godotsharp_color_get_ok_hsl_h));

    internal static float godotsharp_color_get_ok_hsl_s(in Color self) =>
        throw NotAvailable(nameof(godotsharp_color_get_ok_hsl_s));

    internal static float godotsharp_color_get_ok_hsl_l(in Color self) =>
        throw NotAvailable(nameof(godotsharp_color_get_ok_hsl_l));

    internal static Color godotsharp_color_from_ok_hsl(float h, float s, float l, float alpha) =>
        throw NotAvailable(nameof(godotsharp_color_from_ok_hsl));
}
