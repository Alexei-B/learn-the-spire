using System;
using System.Diagnostics.CodeAnalysis;
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

	public int R8
	{
		readonly get
		{
			return (int)Math.Round(R * 255f);
		}
		set
		{
			R = (float)value / 255f;
		}
	}

	public int G8
	{
		readonly get
		{
			return (int)Math.Round(G * 255f);
		}
		set
		{
			G = (float)value / 255f;
		}
	}

	public int B8
	{
		readonly get
		{
			return (int)Math.Round(B * 255f);
		}
		set
		{
			B = (float)value / 255f;
		}
	}

	public int A8
	{
		readonly get
		{
			return (int)Math.Round(A * 255f);
		}
		set
		{
			A = (float)value / 255f;
		}
	}

	public float H
	{
		readonly get
		{
			float num = Math.Max(R, Math.Max(G, B));
			float num2 = Math.Min(R, Math.Min(G, B));
			float num3 = num - num2;
			if (num3 == 0f)
			{
				return 0f;
			}
			float num4 = ((R == num) ? ((G - B) / num3) : ((G != num) ? (4f + (R - G) / num3) : (2f + (B - R) / num3)));
			num4 /= 6f;
			if (num4 < 0f)
			{
				num4 += 1f;
			}
			return num4;
		}
		set
		{
			this = FromHsv(value, S, V, A);
		}
	}

	public float S
	{
		readonly get
		{
			float num = Math.Max(R, Math.Max(G, B));
			float num2 = Math.Min(R, Math.Min(G, B));
			float num3 = num - num2;
			if (num != 0f)
			{
				return num3 / num;
			}
			return 0f;
		}
		set
		{
			this = FromHsv(H, value, V, A);
		}
	}

	public float V
	{
		readonly get
		{
			return Math.Max(R, Math.Max(G, B));
		}
		set
		{
			this = FromHsv(H, S, value, A);
		}
	}

	public float OkHslH
	{
		readonly get
		{
			return NativeFuncs.godotsharp_color_get_ok_hsl_h(in this);
		}
		set
		{
			this = FromOkHsl(value, OkHslS, OkHslL, A);
		}
	}

	public float OkHslS
	{
		readonly get
		{
			return NativeFuncs.godotsharp_color_get_ok_hsl_s(in this);
		}
		set
		{
			this = FromOkHsl(OkHslH, value, OkHslL, A);
		}
	}

	public float OkHslL
	{
		readonly get
		{
			return NativeFuncs.godotsharp_color_get_ok_hsl_l(in this);
		}
		set
		{
			this = FromOkHsl(OkHslH, OkHslS, value, A);
		}
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
				_ => throw new ArgumentOutOfRangeException("index"), 
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
				throw new ArgumentOutOfRangeException("index");
			}
		}
	}

	public readonly Color Blend(Color over)
	{
		float num = 1f - over.A;
		Color result = default(Color);
		result.A = A * num + over.A;
		if (result.A == 0f)
		{
			return new Color(0f, 0f, 0f, 0f);
		}
		result.R = (R * A * num + over.R * over.A) / result.A;
		result.G = (G * A * num + over.G * over.A) / result.A;
		result.B = (B * A * num + over.B * over.A) / result.A;
		return result;
	}

	public readonly Color Clamp(Color? min = null, Color? max = null)
	{
		Color color = min ?? new Color(0f, 0f, 0f, 0f);
		Color color2 = max ?? new Color(1f, 1f, 1f);
		return new Color(Mathf.Clamp(R, color.R, color2.R), Mathf.Clamp(G, color.G, color2.G), Mathf.Clamp(B, color.B, color2.B), Mathf.Clamp(A, color.A, color2.A));
	}

	public readonly Color Darkened(float amount)
	{
		Color result = this;
		result.R *= 1f - amount;
		result.G *= 1f - amount;
		result.B *= 1f - amount;
		return result;
	}

	public readonly Color Inverted()
	{
		return new Color(1f - R, 1f - G, 1f - B, A);
	}

	public readonly Color Lightened(float amount)
	{
		Color result = this;
		result.R += (1f - result.R) * amount;
		result.G += (1f - result.G) * amount;
		result.B += (1f - result.B) * amount;
		return result;
	}

	public readonly Color Lerp(Color to, float weight)
	{
		return new Color(Mathf.Lerp(R, to.R, weight), Mathf.Lerp(G, to.G, weight), Mathf.Lerp(B, to.B, weight), Mathf.Lerp(A, to.A, weight));
	}

	public readonly Color LinearToSrgb()
	{
		return new Color((R < 0.0031308f) ? (12.92f * R) : (1.055f * Mathf.Pow(R, 5f / 12f) - 0.055f), (G < 0.0031308f) ? (12.92f * G) : (1.055f * Mathf.Pow(G, 5f / 12f) - 0.055f), (B < 0.0031308f) ? (12.92f * B) : (1.055f * Mathf.Pow(B, 5f / 12f) - 0.055f), A);
	}

	public readonly Color SrgbToLinear()
	{
		return new Color((R < 0.04045f) ? (R * 0.07739938f) : Mathf.Pow((R + 0.055f) * 0.9478673f, 2.4f), (G < 0.04045f) ? (G * 0.07739938f) : Mathf.Pow((G + 0.055f) * 0.9478673f, 2.4f), (B < 0.04045f) ? (B * 0.07739938f) : Mathf.Pow((B + 0.055f) * 0.9478673f, 2.4f), A);
	}

	public readonly uint ToAbgr32()
	{
		return (uint)(((((((byte)Math.Round(A * 255f) << 8) | (byte)Math.Round(B * 255f)) << 8) | (byte)Math.Round(G * 255f)) << 8) | (byte)Math.Round(R * 255f));
	}

	public readonly ulong ToAbgr64()
	{
		return ((((((ulong)(ushort)Math.Round(A * 65535f) << 16) | (ushort)Math.Round(B * 65535f)) << 16) | (ushort)Math.Round(G * 65535f)) << 16) | (ushort)Math.Round(R * 65535f);
	}

	public readonly uint ToArgb32()
	{
		return (uint)(((((((byte)Math.Round(A * 255f) << 8) | (byte)Math.Round(R * 255f)) << 8) | (byte)Math.Round(G * 255f)) << 8) | (byte)Math.Round(B * 255f));
	}

	public readonly ulong ToArgb64()
	{
		return ((((((ulong)(ushort)Math.Round(A * 65535f) << 16) | (ushort)Math.Round(R * 65535f)) << 16) | (ushort)Math.Round(G * 65535f)) << 16) | (ushort)Math.Round(B * 65535f);
	}

	public readonly uint ToRgba32()
	{
		return (uint)(((((((byte)Math.Round(R * 255f) << 8) | (byte)Math.Round(G * 255f)) << 8) | (byte)Math.Round(B * 255f)) << 8) | (byte)Math.Round(A * 255f));
	}

	public readonly ulong ToRgba64()
	{
		return ((((((ulong)(ushort)Math.Round(R * 65535f) << 16) | (ushort)Math.Round(G * 65535f)) << 16) | (ushort)Math.Round(B * 65535f)) << 16) | (ushort)Math.Round(A * 65535f);
	}

	public readonly string ToHtml(bool includeAlpha = true)
	{
		string empty = string.Empty;
		empty += ToHex32(R);
		empty += ToHex32(G);
		empty += ToHex32(B);
		if (includeAlpha)
		{
			empty += ToHex32(A);
		}
		return empty;
	}

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
		A = (float)(rgba & 0xFF) / 255f;
		rgba >>= 8;
		B = (float)(rgba & 0xFF) / 255f;
		rgba >>= 8;
		G = (float)(rgba & 0xFF) / 255f;
		rgba >>= 8;
		R = (float)(rgba & 0xFF) / 255f;
	}

	public Color(ulong rgba)
	{
		A = (float)(rgba & 0xFFFF) / 65535f;
		rgba >>= 16;
		B = (float)(rgba & 0xFFFF) / 65535f;
		rgba >>= 16;
		G = (float)(rgba & 0xFFFF) / 65535f;
		rgba >>= 16;
		R = (float)(rgba & 0xFFFF) / 65535f;
	}

	public Color(string code)
	{
		if (HtmlIsValid(code))
		{
			this = FromHtml(code);
		}
		else
		{
			this = Named(code);
		}
	}

	public Color(string code, float alpha)
		: this(code)
	{
		A = alpha;
	}

	public static Color FromHtml(ReadOnlySpan<char> rgba)
	{
		Color result = default(Color);
		if (rgba.Length == 0)
		{
			result.R = 0f;
			result.G = 0f;
			result.B = 0f;
			result.A = 1f;
			return result;
		}
		if (rgba[0] == '#')
		{
			rgba = rgba.Slice(1);
		}
		bool num = rgba.Length < 5;
		bool flag;
		if (rgba.Length == 8)
		{
			flag = true;
		}
		else if (rgba.Length == 6)
		{
			flag = false;
		}
		else if (rgba.Length == 4)
		{
			flag = true;
		}
		else
		{
			if (rgba.Length != 3)
			{
				throw new ArgumentOutOfRangeException($"Invalid color code. Length is {rgba.Length}, but a length of 6 or 8 is expected: {rgba}");
			}
			flag = false;
		}
		result.A = 1f;
		if (num)
		{
			result.R = (float)ParseCol4(rgba, 0) / 15f;
			result.G = (float)ParseCol4(rgba, 1) / 15f;
			result.B = (float)ParseCol4(rgba, 2) / 15f;
			if (flag)
			{
				result.A = (float)ParseCol4(rgba, 3) / 15f;
			}
		}
		else
		{
			result.R = (float)ParseCol8(rgba, 0) / 255f;
			result.G = (float)ParseCol8(rgba, 2) / 255f;
			result.B = (float)ParseCol8(rgba, 4) / 255f;
			if (flag)
			{
				result.A = (float)ParseCol8(rgba, 6) / 255f;
			}
		}
		if (result.R < 0f)
		{
			throw new ArgumentOutOfRangeException($"Invalid color code. Red part is not valid hexadecimal: {rgba}");
		}
		if (result.G < 0f)
		{
			throw new ArgumentOutOfRangeException($"Invalid color code. Green part is not valid hexadecimal: {rgba}");
		}
		if (result.B < 0f)
		{
			throw new ArgumentOutOfRangeException($"Invalid color code. Blue part is not valid hexadecimal: {rgba}");
		}
		if (result.A < 0f)
		{
			throw new ArgumentOutOfRangeException($"Invalid color code. Alpha part is not valid hexadecimal: {rgba}");
		}
		return result;
	}

	public static Color Color8(byte r8, byte g8, byte b8, byte a8 = byte.MaxValue)
	{
		return new Color((float)(int)r8 / 255f, (float)(int)g8 / 255f, (float)(int)b8 / 255f, (float)(int)a8 / 255f);
	}

	private static Color Named(string name)
	{
		if (!FindNamedColor(name, out var color))
		{
			throw new ArgumentOutOfRangeException("Invalid Color Name: " + name);
		}
		return color;
	}

	private static Color Named(string name, Color @default)
	{
		if (!FindNamedColor(name, out var color))
		{
			return @default;
		}
		return color;
	}

	private static bool FindNamedColor(string name, out Color color)
	{
		name = name.Replace(" ", string.Empty, StringComparison.Ordinal);
		name = name.Replace("-", string.Empty, StringComparison.Ordinal);
		name = name.Replace("_", string.Empty, StringComparison.Ordinal);
		name = name.Replace("'", string.Empty, StringComparison.Ordinal);
		name = name.Replace(".", string.Empty, StringComparison.Ordinal);
		name = name.ToUpperInvariant();
		return Colors.NamedColors.TryGetValue(name, out color);
	}

	public static Color FromHsv(float hue, float saturation, float value, float alpha = 1f)
	{
		if (saturation == 0f)
		{
			return new Color(value, value, value, alpha);
		}
		hue *= 6f;
		hue %= 6f;
		int num = (int)hue;
		float num2 = hue - (float)num;
		float num3 = value * (1f - saturation);
		float num4 = value * (1f - saturation * num2);
		float num5 = value * (1f - saturation * (1f - num2));
		return num switch
		{
			0 => new Color(value, num5, num3, alpha), 
			1 => new Color(num4, value, num3, alpha), 
			2 => new Color(num3, value, num5, alpha), 
			3 => new Color(num3, num4, value, alpha), 
			4 => new Color(num5, num3, value, alpha), 
			_ => new Color(value, num3, num4, alpha), 
		};
	}

	public readonly void ToHsv(out float hue, out float saturation, out float value)
	{
		float num = Mathf.Max(R, Mathf.Max(G, B));
		float num2 = Mathf.Min(R, Mathf.Min(G, B));
		float num3 = num - num2;
		if (num3 == 0f)
		{
			hue = 0f;
		}
		else
		{
			if (R == num)
			{
				hue = (G - B) / num3;
			}
			else if (G == num)
			{
				hue = 2f + (B - R) / num3;
			}
			else
			{
				hue = 4f + (R - G) / num3;
			}
			hue /= 6f;
			if (hue < 0f)
			{
				hue += 1f;
			}
		}
		if (num == 0f)
		{
			saturation = 0f;
		}
		else
		{
			saturation = 1f - num2 / num;
		}
		value = num;
	}

	private static int ParseCol4(ReadOnlySpan<char> str, int index)
	{
		char c = str[index];
		if (c >= '0' && c <= '9')
		{
			return c - 48;
		}
		if (c >= 'a' && c <= 'f')
		{
			return c + -87;
		}
		if (c >= 'A' && c <= 'F')
		{
			return c + -55;
		}
		return -1;
	}

	private static int ParseCol8(ReadOnlySpan<char> str, int index)
	{
		return ParseCol4(str, index) * 16 + ParseCol4(str, index + 1);
	}

	public static Color FromOkHsl(float hue, float saturation, float lightness, float alpha = 1f)
	{
		return NativeFuncs.godotsharp_color_from_ok_hsl(hue, saturation, lightness, alpha);
	}

	public static Color FromRgbe9995(uint rgbe)
	{
		float num = rgbe & 0x1FF;
		float num2 = (rgbe >> 9) & 0x1FF;
		float num3 = (rgbe >> 18) & 0x1FF;
		float num4 = rgbe >> 27;
		float num5 = Mathf.Pow(2f, num4 - 15f - 9f);
		float r = num * num5;
		float g = num2 * num5;
		float b = num3 * num5;
		return new Color(r, g, b);
	}

	public static Color FromString(string str, Color @default)
	{
		if (HtmlIsValid(str))
		{
			return FromHtml(str);
		}
		return Named(str, @default);
	}

	private static string ToHex32(float val)
	{
		return ((byte)Mathf.RoundToInt(Mathf.Clamp(val * 255f, 0f, 255f))).HexEncode();
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
		int length = color.Length;
		if (length != 3 && length != 4 && length != 6 && length != 8)
		{
			return false;
		}
		for (int i = 0; i < length; i++)
		{
			if (ParseCol4(color, i) == -1)
			{
				return false;
			}
		}
		return true;
	}

	public static Color operator +(Color left, Color right)
	{
		left.R += right.R;
		left.G += right.G;
		left.B += right.B;
		left.A += right.A;
		return left;
	}

	public static Color operator -(Color left, Color right)
	{
		left.R -= right.R;
		left.G -= right.G;
		left.B -= right.B;
		left.A -= right.A;
		return left;
	}

	public static Color operator -(Color color)
	{
		return Colors.White - color;
	}

	public static Color operator *(Color color, float scale)
	{
		color.R *= scale;
		color.G *= scale;
		color.B *= scale;
		color.A *= scale;
		return color;
	}

	public static Color operator *(float scale, Color color)
	{
		color.R *= scale;
		color.G *= scale;
		color.B *= scale;
		color.A *= scale;
		return color;
	}

	public static Color operator *(Color left, Color right)
	{
		left.R *= right.R;
		left.G *= right.G;
		left.B *= right.B;
		left.A *= right.A;
		return left;
	}

	public static Color operator /(Color color, float scale)
	{
		color.R /= scale;
		color.G /= scale;
		color.B /= scale;
		color.A /= scale;
		return color;
	}

	public static Color operator /(Color left, Color right)
	{
		left.R /= right.R;
		left.G /= right.G;
		left.B /= right.B;
		left.A /= right.A;
		return left;
	}

	public static bool operator ==(Color left, Color right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Color left, Color right)
	{
		return !left.Equals(right);
	}

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

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Color other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Color other)
	{
		if (R == other.R && G == other.G && B == other.B)
		{
			return A == other.A;
		}
		return false;
	}

	public readonly bool IsEqualApprox(Color other)
	{
		if (Mathf.IsEqualApprox(R, other.R) && Mathf.IsEqualApprox(G, other.G) && Mathf.IsEqualApprox(B, other.B))
		{
			return Mathf.IsEqualApprox(A, other.A);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(R, G, B, A);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"({R.ToString(format, CultureInfo.InvariantCulture)}, {G.ToString(format, CultureInfo.InvariantCulture)}, {B.ToString(format, CultureInfo.InvariantCulture)}, {A.ToString(format, CultureInfo.InvariantCulture)})";
	}
}
