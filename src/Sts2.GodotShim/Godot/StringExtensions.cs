namespace Godot;

/// <summary>
/// Minimal subset of Godot's StringExtensions. Only the pure-managed members the
/// shimmed value types need are provided; grow this on demand (do NOT pull in the
/// native-backed members, which would cascade into the interop struct surface).
/// Implementations copied verbatim from GodotSharp to preserve exact behavior.
/// </summary>
public static class StringExtensions
{
    internal static string HexEncode(this byte b)
    {
        string text = string.Empty;
        for (int i = 0; i < 2; i++)
        {
            int num = b & 0xF;
            char c = ((num >= 10) ? ((char)(97 + num - 10)) : ((char)(48 + num)));
            b >>= 4;
            text = c + text;
        }
        return text;
    }

    public static string HexEncode(this byte[] bytes)
    {
        string text = string.Empty;
        foreach (byte b in bytes)
        {
            text += b.HexEncode();
        }
        return text;
    }
}
