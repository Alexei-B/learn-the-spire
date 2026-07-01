using System;
using System.Globalization;
using Godot.NativeInterop;

namespace Godot;

[Serializable]
public struct Color : IEquatable<Color>
{
    public float R;
    public float G;
    public float B;
    public float A;

    public Color(float r, float g, float b, float a = 1f)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }

    public Color(Color c, float a = 1f)
    {
        R = c.R;
        G = c.G;
        B = c.B;
        A = a;
    }

    public Color(uint rgba)
    {
        A = (rgba & 0xFF) / 255f;
        rgba >>= 8;
        B = (rgba & 0xFF) / 255f;
        rgba >>= 8;
        G = (rgba & 0xFF) / 255f;
        rgba >>= 8;
        R = (rgba & 0xFF) / 255f;
    }

    public Color(ulong rgba)
    {
        A = (rgba & 0xFFFF) / 65535f;
        rgba >>= 16;
        B = (rgba & 0xFFFF) / 65535f;
        rgba >>= 16;
        G = (rgba & 0xFFFF) / 65535f;
        rgba >>= 16;
        R = (rgba & 0xFFFF) / 65535f;
    }

    public Color(string code)
    {
        this = HtmlIsValid(code) ? FromHtml(code) : Named(code);
    }

    public Color(string code, float alpha)
        : this(code)
    {
        A = alpha;
    }

    public int R8
    {
        readonly get => (int)Math.Round(R * 255f);
        set => R = value / 255f;
    }

    public int G8
    {
        readonly get => (int)Math.Round(G * 255f);
        set => G = value / 255f;
    }

    public int B8
    {
        readonly get => (int)Math.Round(B * 255f);
        set => B = value / 255f;
    }

    public int A8
    {
        readonly get => (int)Math.Round(A * 255f);
        set => A = value / 255f;
    }

    public float H
    {
        readonly get
        {
            float max = Mathf.Max(R, Mathf.Max(G, B));
            float min = Mathf.Min(R, Mathf.Min(G, B));
            float delta = max - min;
            if (delta == 0f)
            {
                return 0f;
            }

            float hue;
            if (R == max)
            {
                hue = (G - B) / delta;
            }
            else if (G == max)
            {
                hue = 2f + (B - R) / delta;
            }
            else
            {
                hue = 4f + (R - G) / delta;
            }

            hue /= 6f;
            if (hue < 0f)
            {
                hue += 1f;
            }

            return hue;
        }
        set => this = FromHsv(value, S, V, A);
    }

    public float S
    {
        readonly get
        {
            float max = Mathf.Max(R, Mathf.Max(G, B));
            float min = Mathf.Min(R, Mathf.Min(G, B));
            return max == 0f ? 0f : (max - min) / max;
        }
        set => this = FromHsv(H, value, V, A);
    }

    public float V
    {
        readonly get => Mathf.Max(R, Mathf.Max(G, B));
        set => this = FromHsv(H, S, value, A);
    }

    public float OkHslH
    {
        readonly get => NativeFuncs.godotsharp_color_get_ok_hsl_h(in this);
        set => this = FromOkHsl(value, OkHslS, OkHslL, A);
    }

    public float OkHslS
    {
        readonly get => NativeFuncs.godotsharp_color_get_ok_hsl_s(in this);
        set => this = FromOkHsl(OkHslH, value, OkHslL, A);
    }

    public float OkHslL
    {
        readonly get => NativeFuncs.godotsharp_color_get_ok_hsl_l(in this);
        set => this = FromOkHsl(OkHslH, OkHslS, value, A);
    }

    public readonly float Luminance => 0.2126f * R + 0.7152f * G + 0.0722f * B;

    public float this[int index]
    {
        readonly get
        {
            return index switch
            {
                0 => R,
                1 => G,
                2 => B,
                3 => A,
                _ => throw new ArgumentOutOfRangeException(nameof(index))
            };
        }
        set
        {
            switch (index)
            {
                case 0:
                    R = value;
                    break;
                case 1:
                    G = value;
                    break;
                case 2:
                    B = value;
                    break;
                case 3:
                    A = value;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
    }

    public readonly Color Blend(Color over)
    {
        float alpha = A * (1f - over.A) + over.A;
        if (alpha == 0f)
        {
            return new Color(0f, 0f, 0f, 0f);
        }

        float inverse = 1f / alpha;
        return new Color(
            (R * A * (1f - over.A) + over.R * over.A) * inverse,
            (G * A * (1f - over.A) + over.G * over.A) * inverse,
            (B * A * (1f - over.A) + over.B * over.A) * inverse,
            alpha);
    }

    public readonly Color Clamp(Color? min = null, Color? max = null)
    {
        Color minColor = min ?? new Color(0f, 0f, 0f, 0f);
        Color maxColor = max ?? new Color(1f, 1f, 1f, 1f);
        return new Color(
            Mathf.Clamp(R, minColor.R, maxColor.R),
            Mathf.Clamp(G, minColor.G, maxColor.G),
            Mathf.Clamp(B, minColor.B, maxColor.B),
            Mathf.Clamp(A, minColor.A, maxColor.A));
    }

    public readonly Color Darkened(float amount)
    {
        return new Color(R * (1f - amount), G * (1f - amount), B * (1f - amount), A);
    }

    public readonly Color Lightened(float amount)
    {
        return new Color(
            R + (1f - R) * amount,
            G + (1f - G) * amount,
            B + (1f - B) * amount,
            A);
    }

    public readonly Color Inverted() => new Color(1f - R, 1f - G, 1f - B, A);

    public readonly Color Lerp(Color to, float weight)
    {
        return new Color(
            Mathf.Lerp(R, to.R, weight),
            Mathf.Lerp(G, to.G, weight),
            Mathf.Lerp(B, to.B, weight),
            Mathf.Lerp(A, to.A, weight));
    }

    public readonly Color LinearToSrgb()
    {
        return new Color(
            LinearToSrgbChannel(R),
            LinearToSrgbChannel(G),
            LinearToSrgbChannel(B),
            A);
    }

    public readonly Color SrgbToLinear()
    {
        return new Color(
            SrgbToLinearChannel(R),
            SrgbToLinearChannel(G),
            SrgbToLinearChannel(B),
            A);
    }

    private static float LinearToSrgbChannel(float channel)
    {
        return channel < 0.0031308f
            ? 12.92f * channel
            : 1.055f * Mathf.Pow(channel, 5f / 12f) - 0.055f;
    }

    private static float SrgbToLinearChannel(float channel)
    {
        return channel < 0.04045f
            ? channel * 0.07739938f
            : Mathf.Pow((channel + 0.055f) * 0.9478673f, 2.4f);
    }

    public readonly uint ToRgba32()
    {
        uint value = (uint)R8;
        value <<= 8;
        value |= (uint)G8;
        value <<= 8;
        value |= (uint)B8;
        value <<= 8;
        value |= (uint)A8;
        return value;
    }

    public readonly uint ToArgb32()
    {
        uint value = (uint)A8;
        value <<= 8;
        value |= (uint)R8;
        value <<= 8;
        value |= (uint)G8;
        value <<= 8;
        value |= (uint)B8;
        return value;
    }

    public readonly uint ToAbgr32()
    {
        uint value = (uint)A8;
        value <<= 8;
        value |= (uint)B8;
        value <<= 8;
        value |= (uint)G8;
        value <<= 8;
        value |= (uint)R8;
        return value;
    }

    public readonly ulong ToRgba64()
    {
        ulong value = Channel16(R);
        value <<= 16;
        value |= Channel16(G);
        value <<= 16;
        value |= Channel16(B);
        value <<= 16;
        value |= Channel16(A);
        return value;
    }

    public readonly ulong ToArgb64()
    {
        ulong value = Channel16(A);
        value <<= 16;
        value |= Channel16(R);
        value <<= 16;
        value |= Channel16(G);
        value <<= 16;
        value |= Channel16(B);
        return value;
    }

    public readonly ulong ToAbgr64()
    {
        ulong value = Channel16(A);
        value <<= 16;
        value |= Channel16(B);
        value <<= 16;
        value |= Channel16(G);
        value <<= 16;
        value |= Channel16(R);
        return value;
    }

    private static ulong Channel16(float channel) => (ulong)Math.Round(channel * 65535f);

    public readonly string ToHtml(bool includeAlpha = true)
    {
        string text = ToHex32(R) + ToHex32(G) + ToHex32(B);
        if (includeAlpha)
        {
            text += ToHex32(A);
        }

        return text;
    }

    public static Color FromHtml(ReadOnlySpan<char> rgba)
    {
        if (rgba.Length > 0 && rgba[0] == '#')
        {
            rgba = rgba.Slice(1);
        }

        float r;
        float g;
        float b;
        float a = 1f;

        switch (rgba.Length)
        {
            case 3:
            case 4:
                r = ParseCol4(rgba, 0) / 15f;
                g = ParseCol4(rgba, 1) / 15f;
                b = ParseCol4(rgba, 2) / 15f;
                if (rgba.Length == 4)
                {
                    a = ParseCol4(rgba, 3) / 15f;
                }

                break;
            case 6:
            case 8:
                r = ParseCol8(rgba, 0) / 255f;
                g = ParseCol8(rgba, 2) / 255f;
                b = ParseCol8(rgba, 4) / 255f;
                if (rgba.Length == 8)
                {
                    a = ParseCol8(rgba, 6) / 255f;
                }

                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(rgba), "Invalid color code length.");
        }

        if (r < 0f || g < 0f || b < 0f || a < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(rgba), "Color code contains invalid characters.");
        }

        return new Color(r, g, b, a);
    }

    private static int ParseCol4(ReadOnlySpan<char> span, int index)
    {
        char c = span[index];
        if (c >= '0' && c <= '9')
        {
            return c - '0';
        }

        if (c >= 'a' && c <= 'f')
        {
            return c - 'a' + 10;
        }

        if (c >= 'A' && c <= 'F')
        {
            return c - 'A' + 10;
        }

        return -1;
    }

    private static int ParseCol8(ReadOnlySpan<char> span, int index)
    {
        return ParseCol4(span, index) * 16 + ParseCol4(span, index + 1);
    }

    public static Color Color8(byte r8, byte g8, byte b8, byte a8 = 255)
    {
        return new Color(r8 / 255f, g8 / 255f, b8 / 255f, a8 / 255f);
    }

    public static bool HtmlIsValid(ReadOnlySpan<char> color)
    {
        if (color.IsEmpty)
        {
            return false;
        }

        if (color[0] == '#')
        {
            color = color.Slice(1);
        }

        if (color.Length != 3 && color.Length != 4 && color.Length != 6 && color.Length != 8)
        {
            return false;
        }

        foreach (char c in color)
        {
            bool isHex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
            if (!isHex)
            {
                return false;
            }
        }

        return true;
    }

    public static Color FromHsv(float hue, float saturation, float value, float alpha = 1f)
    {
        if (saturation == 0f)
        {
            return new Color(value, value, value, alpha);
        }

        float sector = hue * 6f;
        if (sector >= 6f)
        {
            sector = 0f;
        }

        int i = (int)sector;
        float f = sector - i;
        float p = value * (1f - saturation);
        float q = value * (1f - saturation * f);
        float t = value * (1f - saturation * (1f - f));

        float r;
        float g;
        float b;
        switch (i)
        {
            case 0:
                r = value;
                g = t;
                b = p;
                break;
            case 1:
                r = q;
                g = value;
                b = p;
                break;
            case 2:
                r = p;
                g = value;
                b = t;
                break;
            case 3:
                r = p;
                g = q;
                b = value;
                break;
            case 4:
                r = t;
                g = p;
                b = value;
                break;
            default:
                r = value;
                g = p;
                b = q;
                break;
        }

        return new Color(r, g, b, alpha);
    }

    public readonly void ToHsv(out float hue, out float saturation, out float value)
    {
        hue = H;
        saturation = S;
        value = V;
    }

    public static Color FromOkHsl(float hue, float saturation, float lightness, float alpha = 1f)
    {
        return NativeFuncs.godotsharp_color_from_ok_hsl(hue, saturation, lightness, alpha);
    }

    public static Color FromRgbe9995(uint rgbe)
    {
        float r = rgbe & 0x1FF;
        float g = (rgbe >> 9) & 0x1FF;
        float b = (rgbe >> 18) & 0x1FF;
        float exponent = rgbe >> 27;
        float scale = Mathf.Pow(2f, exponent - 15f - 9f);
        return new Color(r * scale, g * scale, b * scale, 1f);
    }

    public static Color FromString(string str, Color @default)
    {
        return HtmlIsValid(str) ? FromHtml(str) : Named(str, @default);
    }

    private static Color Named(string name)
    {
        if (!FindNamedColor(name, out Color color))
        {
            throw new ArgumentOutOfRangeException(nameof(name), "Unknown color name.");
        }

        return color;
    }

    private static Color Named(string name, Color @default)
    {
        return FindNamedColor(name, out Color color) ? color : @default;
    }

    private static bool FindNamedColor(string name, out Color color)
    {
        string key = name
            .Replace(" ", string.Empty, StringComparison.Ordinal)
            .Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace("'", string.Empty, StringComparison.Ordinal)
            .Replace(".", string.Empty, StringComparison.Ordinal)
            .ToUpperInvariant();
        return Colors.NamedColors.TryGetValue(key, out color);
    }

    private static string ToHex32(float val)
    {
        return ((byte)Mathf.RoundToInt(Mathf.Clamp(val * 255f, 0f, 255f))).HexEncode();
    }

    public readonly bool Equals(Color other)
    {
        return R == other.R && G == other.G && B == other.B && A == other.A;
    }

    public readonly bool IsEqualApprox(Color other)
    {
        return Mathf.IsEqualApprox(R, other.R)
            && Mathf.IsEqualApprox(G, other.G)
            && Mathf.IsEqualApprox(B, other.B)
            && Mathf.IsEqualApprox(A, other.A);
    }

    public override readonly bool Equals(object? obj) => obj is Color other && Equals(other);

    public override readonly int GetHashCode() => HashCode.Combine(R, G, B, A);

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"({R.ToString(format, CultureInfo.InvariantCulture)}, {G.ToString(format, CultureInfo.InvariantCulture)}, {B.ToString(format, CultureInfo.InvariantCulture)}, {A.ToString(format, CultureInfo.InvariantCulture)})";
    }

    public static Color operator +(Color left, Color right)
    {
        return new Color(left.R + right.R, left.G + right.G, left.B + right.B, left.A + right.A);
    }

    public static Color operator -(Color left, Color right)
    {
        return new Color(left.R - right.R, left.G - right.G, left.B - right.B, left.A - right.A);
    }

    public static Color operator -(Color color) => Colors.White - color;

    public static Color operator *(Color color, float scale)
    {
        return new Color(color.R * scale, color.G * scale, color.B * scale, color.A * scale);
    }

    public static Color operator *(float scale, Color color)
    {
        return new Color(color.R * scale, color.G * scale, color.B * scale, color.A * scale);
    }

    public static Color operator *(Color left, Color right)
    {
        return new Color(left.R * right.R, left.G * right.G, left.B * right.B, left.A * right.A);
    }

    public static Color operator /(Color color, float divisor)
    {
        return new Color(color.R / divisor, color.G / divisor, color.B / divisor, color.A / divisor);
    }

    public static Color operator /(Color left, Color right)
    {
        return new Color(left.R / right.R, left.G / right.G, left.B / right.B, left.A / right.A);
    }

    public static bool operator ==(Color left, Color right) => left.Equals(right);

    public static bool operator !=(Color left, Color right) => !left.Equals(right);

    public static bool operator <(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                {
                    return left.A < right.A;
                }

                return left.B < right.B;
            }

            return left.G < right.G;
        }

        return left.R < right.R;
    }

    public static bool operator >(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                {
                    return left.A > right.A;
                }

                return left.B > right.B;
            }

            return left.G > right.G;
        }

        return left.R > right.R;
    }

    public static bool operator <=(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                {
                    return left.A <= right.A;
                }

                return left.B < right.B;
            }

            return left.G < right.G;
        }

        return left.R < right.R;
    }

    public static bool operator >=(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                {
                    return left.A >= right.A;
                }

                return left.B > right.B;
            }

            return left.G > right.G;
        }

        return left.R > right.R;
    }
}
