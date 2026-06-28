using System;
using System.Runtime.CompilerServices;

namespace Godot;

public static class Mathf
{
	public const float Tau = (float)Math.PI * 2f;

	public const float Pi = (float)Math.PI;

	public const float Inf = float.PositiveInfinity;

	public const float NaN = float.NaN;

	private const float DegToRadConstF = (float)Math.PI / 180f;

	private const double DegToRadConstD = Math.PI / 180.0;

	private const float RadToDegConstF = 57.29578f;

	private const double RadToDegConstD = 57.295779513082316;

	public const float E = (float)Math.E;

	public const float Sqrt2 = 1.4142135f;

	private const float EpsilonF = 1E-06f;

	private const double EpsilonD = 1E-14;

	public const float Epsilon = 1E-06f;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Abs(int s)
	{
		return Math.Abs(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Abs(float s)
	{
		return Math.Abs(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Abs(double s)
	{
		return Math.Abs(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Acos(float s)
	{
		return MathF.Acos(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Acos(double s)
	{
		return Math.Acos(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Acosh(float s)
	{
		return MathF.Acosh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Acosh(double s)
	{
		return Math.Acosh(s);
	}

	public static float AngleDifference(float from, float to)
	{
		float num = (to - from) % ((float)Math.PI * 2f);
		return 2f * num % ((float)Math.PI * 2f) - num;
	}

	public static double AngleDifference(double from, double to)
	{
		double num = (to - from) % (Math.PI * 2.0);
		return 2.0 * num % (Math.PI * 2.0) - num;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Asin(float s)
	{
		return MathF.Asin(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Asin(double s)
	{
		return Math.Asin(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Asinh(float s)
	{
		return MathF.Asinh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Asinh(double s)
	{
		return Math.Asinh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Atan(float s)
	{
		return MathF.Atan(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Atan(double s)
	{
		return Math.Atan(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Atan2(float y, float x)
	{
		return MathF.Atan2(y, x);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Atan2(double y, double x)
	{
		return Math.Atan2(y, x);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Atanh(float s)
	{
		return MathF.Atanh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Atanh(double s)
	{
		return Math.Atanh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Ceil(float s)
	{
		return MathF.Ceiling(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Ceil(double s)
	{
		return Math.Ceiling(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Clamp(int value, int min, int max)
	{
		return Math.Clamp(value, min, max);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Clamp(float value, float min, float max)
	{
		return Math.Clamp(value, min, max);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Clamp(double value, double min, double max)
	{
		return Math.Clamp(value, min, max);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Cos(float s)
	{
		return MathF.Cos(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Cos(double s)
	{
		return Math.Cos(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Cosh(float s)
	{
		return MathF.Cosh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Cosh(double s)
	{
		return Math.Cosh(s);
	}

	public static float CubicInterpolate(float from, float to, float pre, float post, float weight)
	{
		return 0.5f * (from * 2f + (0f - pre + to) * weight + (2f * pre - 5f * from + 4f * to - post) * (weight * weight) + (0f - pre + 3f * from - 3f * to + post) * (weight * weight * weight));
	}

	public static double CubicInterpolate(double from, double to, double pre, double post, double weight)
	{
		return 0.5 * (from * 2.0 + (0.0 - pre + to) * weight + (2.0 * pre - 5.0 * from + 4.0 * to - post) * (weight * weight) + (0.0 - pre + 3.0 * from - 3.0 * to + post) * (weight * weight * weight));
	}

	public static float CubicInterpolateAngle(float from, float to, float pre, float post, float weight)
	{
		float num = from % ((float)Math.PI * 2f);
		float num2 = (pre - num) % ((float)Math.PI * 2f);
		float pre2 = num + 2f * num2 % ((float)Math.PI * 2f) - num2;
		float num3 = (to - num) % ((float)Math.PI * 2f);
		float num4 = num + 2f * num3 % ((float)Math.PI * 2f) - num3;
		float num5 = (post - num4) % ((float)Math.PI * 2f);
		float post2 = num4 + 2f * num5 % ((float)Math.PI * 2f) - num5;
		return CubicInterpolate(num, num4, pre2, post2, weight);
	}

	public static double CubicInterpolateAngle(double from, double to, double pre, double post, double weight)
	{
		double num = from % (Math.PI * 2.0);
		double num2 = (pre - num) % (Math.PI * 2.0);
		double pre2 = num + 2.0 * num2 % (Math.PI * 2.0) - num2;
		double num3 = (to - num) % (Math.PI * 2.0);
		double num4 = num + 2.0 * num3 % (Math.PI * 2.0) - num3;
		double num5 = (post - num4) % (Math.PI * 2.0);
		double post2 = num4 + 2.0 * num5 % (Math.PI * 2.0) - num5;
		return CubicInterpolate(num, num4, pre2, post2, weight);
	}

	public static float CubicInterpolateInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
	{
		float num = Lerp(0f, toT, weight);
		float num2 = Lerp(pre, from, (preT == 0f) ? 0f : ((num - preT) / (0f - preT)));
		float num3 = Lerp(from, to, (toT == 0f) ? 0.5f : (num / toT));
		float to2 = Lerp(to, post, (postT - toT == 0f) ? 1f : ((num - toT) / (postT - toT)));
		float num4 = Lerp(num2, num3, (toT - preT == 0f) ? 0f : ((num - preT) / (toT - preT)));
		float to3 = Lerp(num3, to2, (postT == 0f) ? 1f : (num / postT));
		return Lerp(num4, to3, (toT == 0f) ? 0.5f : (num / toT));
	}

	public static double CubicInterpolateInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
	{
		double num = Lerp(0.0, toT, weight);
		double num2 = Lerp(pre, from, (preT == 0.0) ? 0.0 : ((num - preT) / (0.0 - preT)));
		double num3 = Lerp(from, to, (toT == 0.0) ? 0.5 : (num / toT));
		double to2 = Lerp(to, post, (postT - toT == 0.0) ? 1.0 : ((num - toT) / (postT - toT)));
		double num4 = Lerp(num2, num3, (toT - preT == 0.0) ? 0.0 : ((num - preT) / (toT - preT)));
		double to3 = Lerp(num3, to2, (postT == 0.0) ? 1.0 : (num / postT));
		return Lerp(num4, to3, (toT == 0.0) ? 0.5 : (num / toT));
	}

	public static float CubicInterpolateAngleInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
	{
		float num = from % ((float)Math.PI * 2f);
		float num2 = (pre - num) % ((float)Math.PI * 2f);
		float pre2 = num + 2f * num2 % ((float)Math.PI * 2f) - num2;
		float num3 = (to - num) % ((float)Math.PI * 2f);
		float num4 = num + 2f * num3 % ((float)Math.PI * 2f) - num3;
		float num5 = (post - num4) % ((float)Math.PI * 2f);
		float post2 = num4 + 2f * num5 % ((float)Math.PI * 2f) - num5;
		return CubicInterpolateInTime(num, num4, pre2, post2, weight, toT, preT, postT);
	}

	public static double CubicInterpolateAngleInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
	{
		double num = from % (Math.PI * 2.0);
		double num2 = (pre - num) % (Math.PI * 2.0);
		double pre2 = num + 2.0 * num2 % (Math.PI * 2.0) - num2;
		double num3 = (to - num) % (Math.PI * 2.0);
		double num4 = num + 2.0 * num3 % (Math.PI * 2.0) - num3;
		double num5 = (post - num4) % (Math.PI * 2.0);
		double post2 = num4 + 2.0 * num5 % (Math.PI * 2.0) - num5;
		return CubicInterpolateInTime(num, num4, pre2, post2, weight, toT, preT, postT);
	}

	public static float BezierInterpolate(float start, float control1, float control2, float end, float t)
	{
		float num = 1f - t;
		float num2 = num * num;
		float num3 = num2 * num;
		float num4 = t * t;
		float num5 = num4 * t;
		return start * num3 + control1 * num2 * t * 3f + control2 * num * num4 * 3f + end * num5;
	}

	public static double BezierInterpolate(double start, double control1, double control2, double end, double t)
	{
		double num = 1.0 - t;
		double num2 = num * num;
		double num3 = num2 * num;
		double num4 = t * t;
		double num5 = num4 * t;
		return start * num3 + control1 * num2 * t * 3.0 + control2 * num * num4 * 3.0 + end * num5;
	}

	public static float BezierDerivative(float start, float control1, float control2, float end, float t)
	{
		float num = 1f - t;
		float num2 = num * num;
		float num3 = t * t;
		return (control1 - start) * 3f * num2 + (control2 - control1) * 6f * num * t + (end - control2) * 3f * num3;
	}

	public static double BezierDerivative(double start, double control1, double control2, double end, double t)
	{
		double num = 1.0 - t;
		double num2 = num * num;
		double num3 = t * t;
		return (control1 - start) * 3.0 * num2 + (control2 - control1) * 6.0 * num * t + (end - control2) * 3.0 * num3;
	}

	public static float DbToLinear(float db)
	{
		return MathF.Exp(db * 0.115129255f);
	}

	public static double DbToLinear(double db)
	{
		return Math.Exp(db * 0.11512925464970228);
	}

	public static float DegToRad(float deg)
	{
		return deg * ((float)Math.PI / 180f);
	}

	public static double DegToRad(double deg)
	{
		return deg * (Math.PI / 180.0);
	}

	public static float Ease(float s, float curve)
	{
		if (s < 0f)
		{
			s = 0f;
		}
		else if (s > 1f)
		{
			s = 1f;
		}
		if (curve > 0f)
		{
			if (curve < 1f)
			{
				return 1f - MathF.Pow(1f - s, 1f / curve);
			}
			return MathF.Pow(s, curve);
		}
		if (curve < 0f)
		{
			if (s < 0.5f)
			{
				return MathF.Pow(s * 2f, 0f - curve) * 0.5f;
			}
			return (1f - MathF.Pow(1f - (s - 0.5f) * 2f, 0f - curve)) * 0.5f + 0.5f;
		}
		return 0f;
	}

	public static double Ease(double s, double curve)
	{
		if (s < 0.0)
		{
			s = 0.0;
		}
		else if (s > 1.0)
		{
			s = 1.0;
		}
		if (curve > 0.0)
		{
			if (curve < 1.0)
			{
				return 1.0 - Math.Pow(1.0 - s, 1.0 / curve);
			}
			return Math.Pow(s, curve);
		}
		if (curve < 0.0)
		{
			if (s < 0.5)
			{
				return Math.Pow(s * 2.0, 0.0 - curve) * 0.5;
			}
			return (1.0 - Math.Pow(1.0 - (s - 0.5) * 2.0, 0.0 - curve)) * 0.5 + 0.5;
		}
		return 0.0;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Exp(float s)
	{
		return MathF.Exp(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Exp(double s)
	{
		return Math.Exp(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Floor(float s)
	{
		return MathF.Floor(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Floor(double s)
	{
		return Math.Floor(s);
	}

	public static float InverseLerp(float from, float to, float weight)
	{
		return (weight - from) / (to - from);
	}

	public static double InverseLerp(double from, double to, double weight)
	{
		return (weight - from) / (to - from);
	}

	public static bool IsEqualApprox(float a, float b)
	{
		if (a == b)
		{
			return true;
		}
		float num = 1E-06f * Math.Abs(a);
		if (num < 1E-06f)
		{
			num = 1E-06f;
		}
		return Math.Abs(a - b) < num;
	}

	public static bool IsEqualApprox(double a, double b)
	{
		if (a == b)
		{
			return true;
		}
		double num = 1E-14 * Math.Abs(a);
		if (num < 1E-14)
		{
			num = 1E-14;
		}
		return Math.Abs(a - b) < num;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsFinite(float s)
	{
		return float.IsFinite(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsFinite(double s)
	{
		return double.IsFinite(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsInf(float s)
	{
		return float.IsInfinity(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsInf(double s)
	{
		return double.IsInfinity(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsNaN(float s)
	{
		return float.IsNaN(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsNaN(double s)
	{
		return double.IsNaN(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsZeroApprox(float s)
	{
		return Math.Abs(s) < 1E-06f;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsZeroApprox(double s)
	{
		return Math.Abs(s) < 1E-14;
	}

	public static float Lerp(float from, float to, float weight)
	{
		return from + (to - from) * weight;
	}

	public static double Lerp(double from, double to, double weight)
	{
		return from + (to - from) * weight;
	}

	public static float LerpAngle(float from, float to, float weight)
	{
		return from + AngleDifference(from, to) * weight;
	}

	public static double LerpAngle(double from, double to, double weight)
	{
		return from + AngleDifference(from, to) * weight;
	}

	public static float LinearToDb(float linear)
	{
		return MathF.Log(linear) * 8.685889f;
	}

	public static double LinearToDb(double linear)
	{
		return Math.Log(linear) * 8.685889638065037;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Log(float s)
	{
		return MathF.Log(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Log(double s)
	{
		return Math.Log(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Max(int a, int b)
	{
		return Math.Max(a, b);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Max(float a, float b)
	{
		return Math.Max(a, b);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Max(double a, double b)
	{
		return Math.Max(a, b);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Min(int a, int b)
	{
		return Math.Min(a, b);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Min(float a, float b)
	{
		return Math.Min(a, b);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Min(double a, double b)
	{
		return Math.Min(a, b);
	}

	public static float MoveToward(float from, float to, float delta)
	{
		if (Math.Abs(to - from) <= delta)
		{
			return to;
		}
		return from + (float)Math.Sign(to - from) * delta;
	}

	public static double MoveToward(double from, double to, double delta)
	{
		if (Math.Abs(to - from) <= delta)
		{
			return to;
		}
		return from + (double)Math.Sign(to - from) * delta;
	}

	public static int NearestPo2(int value)
	{
		value--;
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		value++;
		return value;
	}

	public static int PosMod(int a, int b)
	{
		int num = a % b;
		if ((num < 0 && b > 0) || (num > 0 && b < 0))
		{
			num += b;
		}
		return num;
	}

	public static float PosMod(float a, float b)
	{
		float num = a % b;
		if ((num < 0f && b > 0f) || (num > 0f && b < 0f))
		{
			num += b;
		}
		return num;
	}

	public static double PosMod(double a, double b)
	{
		double num = a % b;
		if ((num < 0.0 && b > 0.0) || (num > 0.0 && b < 0.0))
		{
			num += b;
		}
		return num;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Pow(float x, float y)
	{
		return MathF.Pow(x, y);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Pow(double x, double y)
	{
		return Math.Pow(x, y);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float RadToDeg(float rad)
	{
		return rad * 57.29578f;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double RadToDeg(double rad)
	{
		return rad * 57.295779513082316;
	}

	public static float Remap(float value, float inFrom, float inTo, float outFrom, float outTo)
	{
		return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
	}

	public static double Remap(double value, double inFrom, double inTo, double outFrom, double outTo)
	{
		return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
	}

	public static float RotateToward(float from, float to, float delta)
	{
		float num = AngleDifference(from, to);
		float num2 = Math.Abs(num);
		return from + Math.Clamp(delta, num2 - (float)Math.PI, num2) * ((num >= 0f) ? 1f : (-1f));
	}

	public static double RotateToward(double from, double to, double delta)
	{
		double num = AngleDifference(from, to);
		double num2 = Math.Abs(num);
		return from + Math.Clamp(delta, num2 - Math.PI, num2) * ((num >= 0.0) ? 1.0 : (-1.0));
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Round(float s)
	{
		return MathF.Round(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Round(double s)
	{
		return Math.Round(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Sign(int s)
	{
		return Math.Sign(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Sign(float s)
	{
		return Math.Sign(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Sign(double s)
	{
		return Math.Sign(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Sin(float s)
	{
		return MathF.Sin(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Sin(double s)
	{
		return Math.Sin(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Sinh(float s)
	{
		return MathF.Sinh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Sinh(double s)
	{
		return Math.Sinh(s);
	}

	public static float SmoothStep(float from, float to, float weight)
	{
		if (IsEqualApprox(from, to))
		{
			return from;
		}
		float num = Math.Clamp((weight - from) / (to - from), 0f, 1f);
		return num * num * (3f - 2f * num);
	}

	public static double SmoothStep(double from, double to, double weight)
	{
		if (IsEqualApprox(from, to))
		{
			return from;
		}
		double num = Math.Clamp((weight - from) / (to - from), 0.0, 1.0);
		return num * num * (3.0 - 2.0 * num);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Sqrt(float s)
	{
		return MathF.Sqrt(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Sqrt(double s)
	{
		return Math.Sqrt(s);
	}

	public static int StepDecimals(double step)
	{
		ReadOnlySpan<double> readOnlySpan = new double[9] { 0.9999, 0.09999, 0.009999, 0.0009999, 9.999E-05, 9.999E-06, 9.999E-07, 9.999E-08, 9.999E-09 };
		double num = Math.Abs(step);
		double num2 = num - (double)(int)num;
		for (int i = 0; i < readOnlySpan.Length; i++)
		{
			if (num2 >= readOnlySpan[i])
			{
				return i;
			}
		}
		return 0;
	}

	public static float Snapped(float s, float step)
	{
		if (step != 0f)
		{
			return MathF.Floor(s / step + 0.5f) * step;
		}
		return s;
	}

	public static double Snapped(double s, double step)
	{
		if (step != 0.0)
		{
			return Math.Floor(s / step + 0.5) * step;
		}
		return s;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Tan(float s)
	{
		return MathF.Tan(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Tan(double s)
	{
		return Math.Tan(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static float Tanh(float s)
	{
		return MathF.Tanh(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static double Tanh(double s)
	{
		return Math.Tanh(s);
	}

	public static int Wrap(int value, int min, int max)
	{
		int num = max - min;
		if (num == 0)
		{
			return min;
		}
		return min + ((value - min) % num + num) % num;
	}

	public static float Wrap(float value, float min, float max)
	{
		float num = max - min;
		if (IsZeroApprox(num))
		{
			return min;
		}
		return min + ((value - min) % num + num) % num;
	}

	public static double Wrap(double value, double min, double max)
	{
		double num = max - min;
		if (IsZeroApprox(num))
		{
			return min;
		}
		return min + ((value - min) % num + num) % num;
	}

	public static float PingPong(float value, float length)
	{
		if (length == 0f)
		{
			return 0f;
		}
		return Math.Abs(Fract((value - length) / (length * 2f)) * length * 2f - length);
		static float Fract(float num)
		{
			return num - MathF.Floor(num);
		}
	}

	public static double PingPong(double value, double length)
	{
		if (length == 0.0)
		{
			return 0.0;
		}
		return Math.Abs(Fract((value - length) / (length * 2.0)) * length * 2.0 - length);
		static double Fract(double num)
		{
			return num - Math.Floor(num);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int DecimalCount(double s)
	{
		return DecimalCount((decimal)s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int DecimalCount(decimal s)
	{
		return BitConverter.GetBytes(decimal.GetBits(s)[3])[2];
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int CeilToInt(float s)
	{
		return (int)MathF.Ceiling(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int CeilToInt(double s)
	{
		return (int)Math.Ceiling(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int FloorToInt(float s)
	{
		return (int)MathF.Floor(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int FloorToInt(double s)
	{
		return (int)Math.Floor(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int RoundToInt(float s)
	{
		return (int)MathF.Round(s);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int RoundToInt(double s)
	{
		return (int)Math.Round(s);
	}

	public static (float Sin, float Cos) SinCos(float s)
	{
		return MathF.SinCos(s);
	}

	public static (double Sin, double Cos) SinCos(double s)
	{
		return Math.SinCos(s);
	}

	public static bool IsEqualApprox(float a, float b, float tolerance)
	{
		if (a == b)
		{
			return true;
		}
		return Math.Abs(a - b) < tolerance;
	}

	public static bool IsEqualApprox(double a, double b, double tolerance)
	{
		if (a == b)
		{
			return true;
		}
		return Math.Abs(a - b) < tolerance;
	}
}
