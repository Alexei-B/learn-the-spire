using System.Text;

namespace Godot;

public static class StringExtensions
{
    internal static string HexEncode(this byte b)
    {
        return b.ToString("x2", System.Globalization.CultureInfo.InvariantCulture);
    }

    public static string HexEncode(this byte[] bytes)
    {
        StringBuilder builder = new StringBuilder(bytes.Length * 2);
        foreach (byte b in bytes)
        {
            builder.Append(b.HexEncode());
        }

        return builder.ToString();
    }
}
