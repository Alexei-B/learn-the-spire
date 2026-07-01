using System;
using System.Runtime.CompilerServices;

namespace Godot;

public static class Mathf
{
    public const float Tau = MathF.PI * 2f;
    public const float Pi = MathF.PI;
    public const float Inf = float.PositiveInfinity;
    public const float NaN = float.NaN;
    public const float E = MathF.E;
    public const float Sqrt2 = 1.4142135f;
    public const float Epsilon = 1E-06f;

    private const double EpsilonD = 1E-14;
    private const double TauD = Math.PI * 2.0;

    private const float DegToRadFactor = MathF.PI / 180f;
    private const double DegToRadFactorD = Math.PI / 180.0;

    // ---- trigonometry / basic wrappers ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Abs(int s) => Math.Abs(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Abs(float s) => MathF.Abs(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Abs(double s) => Math.Abs(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Acos(float s) => MathF.Acos(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Acos(double s) => Math.Acos(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Acosh(float s) => MathF.Acosh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Acosh(double s) => Math.Acosh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Asin(float s) => MathF.Asin(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Asin(double s) => Math.Asin(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Asinh(float s) => MathF.Asinh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Asinh(double s) => Math.Asinh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Atan(float s) => MathF.Atan(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Atan(double s) => Math.Atan(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Atan2(float y, float x) => MathF.Atan2(y, x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Atan2(double y, double x) => Math.Atan2(y, x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Atanh(float s) => MathF.Atanh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Atanh(double s) => Math.Atanh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Ceil(float s) => MathF.Ceiling(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Ceil(double s) => Math.Ceiling(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Cos(float s) => MathF.Cos(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Cos(double s) => Math.Cos(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Cosh(float s) => MathF.Cosh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Cosh(double s) => Math.Cosh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp(float s) => MathF.Exp(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Exp(double s) => Math.Exp(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Floor(float s) => MathF.Floor(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Floor(double s) => Math.Floor(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Log(float s) => MathF.Log(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Log(double s) => Math.Log(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Pow(float x, float y) => MathF.Pow(x, y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Pow(double x, double y) => Math.Pow(x, y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Round(float s) => MathF.Round(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Round(double s) => Math.Round(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sin(float s) => MathF.Sin(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sin(double s) => Math.Sin(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sinh(float s) => MathF.Sinh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sinh(double s) => Math.Sinh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sqrt(float s) => MathF.Sqrt(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sqrt(double s) => Math.Sqrt(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Tan(float s) => MathF.Tan(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Tan(double s) => Math.Tan(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Tanh(float s) => MathF.Tanh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Tanh(double s) => Math.Tanh(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float Sin, float Cos) SinCos(float s) => MathF.SinCos(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (double Sin, double Cos) SinCos(double s) => Math.SinCos(s);

    // ---- clamp / min / max / sign ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Clamp(int value, int min, int max) => Math.Clamp(value, min, max);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clamp(float value, float min, float max) => Math.Clamp(value, min, max);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Clamp(double value, double min, double max) => Math.Clamp(value, min, max);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Max(int a, int b) => Math.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(float a, float b) => Math.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Max(double a, double b) => Math.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Min(int a, int b) => Math.Min(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(float a, float b) => Math.Min(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Min(double a, double b) => Math.Min(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Sign(int s) => Math.Sign(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Sign(float s) => Math.Sign(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Sign(double s) => Math.Sign(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsFinite(float s) => float.IsFinite(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsFinite(double s) => double.IsFinite(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsInf(float s) => float.IsInfinity(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsInf(double s) => double.IsInfinity(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsNaN(float s) => float.IsNaN(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsNaN(double s) => double.IsNaN(s);

    // ---- angle helpers ----

    public static float AngleDifference(float from, float to)
    {
        float difference = (to - from) % Tau;
        return 2f * difference % Tau - difference;
    }

    public static double AngleDifference(double from, double to)
    {
        double difference = (to - from) % TauD;
        return 2.0 * difference % TauD - difference;
    }

    // ---- interpolation ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Lerp(float from, float to, float weight) => from + (to - from) * weight;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Lerp(double from, double to, double weight) => from + (to - from) * weight;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float InverseLerp(float from, float to, float weight) => (weight - from) / (to - from);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double InverseLerp(double from, double to, double weight) => (weight - from) / (to - from);

    public static float LerpAngle(float from, float to, float weight)
    {
        return from + AngleDifference(from, to) * weight;
    }

    public static double LerpAngle(double from, double to, double weight)
    {
        return from + AngleDifference(from, to) * weight;
    }

    public static float Remap(float value, float inFrom, float inTo, float outFrom, float outTo)
    {
        return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
    }

    public static double Remap(double value, double inFrom, double inTo, double outFrom, double outTo)
    {
        return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
    }

    public static float CubicInterpolate(float from, float to, float pre, float post, float weight)
    {
        return 0.5f * (2f * from
            + (-pre + to) * weight
            + (2f * pre - 5f * from + 4f * to - post) * (weight * weight)
            + (-pre + 3f * from - 3f * to + post) * (weight * weight * weight));
    }

    public static double CubicInterpolate(double from, double to, double pre, double post, double weight)
    {
        return 0.5 * (2.0 * from
            + (-pre + to) * weight
            + (2.0 * pre - 5.0 * from + 4.0 * to - post) * (weight * weight)
            + (-pre + 3.0 * from - 3.0 * to + post) * (weight * weight * weight));
    }

    public static float CubicInterpolateAngle(float from, float to, float pre, float post, float weight)
    {
        float fromRot = from % Tau;
        float preRot = fromRot + AngleDifference(fromRot, pre);
        float toRot = fromRot + AngleDifference(fromRot, to);
        float postRot = toRot + AngleDifference(toRot, post);
        return CubicInterpolate(fromRot, toRot, preRot, postRot, weight);
    }

    public static double CubicInterpolateAngle(double from, double to, double pre, double post, double weight)
    {
        double fromRot = from % TauD;
        double preRot = fromRot + AngleDifference(fromRot, pre);
        double toRot = fromRot + AngleDifference(fromRot, to);
        double postRot = toRot + AngleDifference(toRot, post);
        return CubicInterpolate(fromRot, toRot, preRot, postRot, weight);
    }

    public static float CubicInterpolateInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
    {
        float t = Lerp(0f, toT, weight);
        float a1 = Lerp(pre, from, preT == 0f ? 0f : (t - preT) / -preT);
        float a2 = Lerp(from, to, toT == 0f ? 0.5f : t / toT);
        float a3 = Lerp(to, post, postT - toT == 0f ? 1f : (t - toT) / (postT - toT));
        float b1 = Lerp(a1, a2, toT - preT == 0f ? 0f : (t - preT) / (toT - preT));
        float b2 = Lerp(a2, a3, postT == 0f ? 1f : t / postT);
        return Lerp(b1, b2, toT == 0f ? 0.5f : t / toT);
    }

    public static double CubicInterpolateInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
    {
        double t = Lerp(0.0, toT, weight);
        double a1 = Lerp(pre, from, preT == 0.0 ? 0.0 : (t - preT) / -preT);
        double a2 = Lerp(from, to, toT == 0.0 ? 0.5 : t / toT);
        double a3 = Lerp(to, post, postT - toT == 0.0 ? 1.0 : (t - toT) / (postT - toT));
        double b1 = Lerp(a1, a2, toT - preT == 0.0 ? 0.0 : (t - preT) / (toT - preT));
        double b2 = Lerp(a2, a3, postT == 0.0 ? 1.0 : t / postT);
        return Lerp(b1, b2, toT == 0.0 ? 0.5 : t / toT);
    }

    public static float CubicInterpolateAngleInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
    {
        float fromRot = from % Tau;
        float preRot = fromRot + AngleDifference(fromRot, pre);
        float toRot = fromRot + AngleDifference(fromRot, to);
        float postRot = toRot + AngleDifference(toRot, post);
        return CubicInterpolateInTime(fromRot, toRot, preRot, postRot, weight, toT, preT, postT);
    }

    public static double CubicInterpolateAngleInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
    {
        double fromRot = from % TauD;
        double preRot = fromRot + AngleDifference(fromRot, pre);
        double toRot = fromRot + AngleDifference(fromRot, to);
        double postRot = toRot + AngleDifference(toRot, post);
        return CubicInterpolateInTime(fromRot, toRot, preRot, postRot, weight, toT, preT, postT);
    }

    public static float BezierInterpolate(float start, float control1, float control2, float end, float t)
    {
        float omt = 1f - t;
        float omt2 = omt * omt;
        float t2 = t * t;
        return start * (omt2 * omt)
            + control1 * (3f * omt2 * t)
            + control2 * (3f * omt * t2)
            + end * (t2 * t);
    }

    public static double BezierInterpolate(double start, double control1, double control2, double end, double t)
    {
        double omt = 1.0 - t;
        double omt2 = omt * omt;
        double t2 = t * t;
        return start * (omt2 * omt)
            + control1 * (3.0 * omt2 * t)
            + control2 * (3.0 * omt * t2)
            + end * (t2 * t);
    }

    public static float BezierDerivative(float start, float control1, float control2, float end, float t)
    {
        float omt = 1f - t;
        return 3f * (omt * omt) * (control1 - start)
            + 6f * omt * t * (control2 - control1)
            + 3f * (t * t) * (end - control2);
    }

    public static double BezierDerivative(double start, double control1, double control2, double end, double t)
    {
        double omt = 1.0 - t;
        return 3.0 * (omt * omt) * (control1 - start)
            + 6.0 * omt * t * (control2 - control1)
            + 3.0 * (t * t) * (end - control2);
    }

    // ---- audio / angle conversions ----

    public static float DbToLinear(float db) => Exp(db * 0.115129255f);

    public static double DbToLinear(double db) => Exp(db * 0.11512925464970228);

    public static float LinearToDb(float linear) => Log(linear) * 8.685889f;

    public static double LinearToDb(double linear) => Log(linear) * 8.685889638065037;

    public static float DegToRad(float deg) => deg * DegToRadFactor;

    public static double DegToRad(double deg) => deg * DegToRadFactorD;

    public static float RadToDeg(float rad) => rad * 57.29578f;

    public static double RadToDeg(double rad) => rad * 57.295779513082316;

    // ---- easing ----

    public static float Ease(float s, float curve)
    {
        s = Clamp(s, 0f, 1f);
        if (curve > 0f)
        {
            return curve < 1f ? 1f - Pow(1f - s, 1f / curve) : Pow(s, curve);
        }

        if (curve < 0f)
        {
            return s < 0.5f
                ? Pow(s * 2f, -curve) * 0.5f
                : (1f - Pow(1f - (s - 0.5f) * 2f, -curve)) * 0.5f + 0.5f;
        }

        return 0f;
    }

    public static double Ease(double s, double curve)
    {
        s = Clamp(s, 0.0, 1.0);
        if (curve > 0.0)
        {
            return curve < 1.0 ? 1.0 - Pow(1.0 - s, 1.0 / curve) : Pow(s, curve);
        }

        if (curve < 0.0)
        {
            return s < 0.5
                ? Pow(s * 2.0, -curve) * 0.5
                : (1.0 - Pow(1.0 - (s - 0.5) * 2.0, -curve)) * 0.5 + 0.5;
        }

        return 0.0;
    }

    // ---- approximate comparisons ----

    public static bool IsEqualApprox(float a, float b)
    {
        if (a == b)
        {
            return true;
        }

        float tolerance = Epsilon * Abs(a);
        if (tolerance < Epsilon)
        {
            tolerance = Epsilon;
        }

        return Abs(a - b) < tolerance;
    }

    public static bool IsEqualApprox(double a, double b)
    {
        if (a == b)
        {
            return true;
        }

        double tolerance = EpsilonD * Abs(a);
        if (tolerance < EpsilonD)
        {
            tolerance = EpsilonD;
        }

        return Abs(a - b) < tolerance;
    }

    public static bool IsEqualApprox(float a, float b, float tolerance)
    {
        return a == b || Abs(a - b) < tolerance;
    }

    public static bool IsEqualApprox(double a, double b, double tolerance)
    {
        return a == b || Abs(a - b) < tolerance;
    }

    public static bool IsZeroApprox(float s) => Abs(s) < Epsilon;

    public static bool IsZeroApprox(double s) => Abs(s) < EpsilonD;

    // ---- movement ----

    public static float MoveToward(float from, float to, float delta)
    {
        return Abs(to - from) <= delta ? to : from + Sign(to - from) * delta;
    }

    public static double MoveToward(double from, double to, double delta)
    {
        return Abs(to - from) <= delta ? to : from + Sign(to - from) * delta;
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
        int r = a % b;
        if ((r < 0 && b > 0) || (r > 0 && b < 0))
        {
            r += b;
        }

        return r;
    }

    public static float PosMod(float a, float b)
    {
        float r = a % b;
        if ((r < 0f && b > 0f) || (r > 0f && b < 0f))
        {
            r += b;
        }

        return r;
    }

    public static double PosMod(double a, double b)
    {
        double r = a % b;
        if ((r < 0.0 && b > 0.0) || (r > 0.0 && b < 0.0))
        {
            r += b;
        }

        return r;
    }

    public static float RotateToward(float from, float to, float delta)
    {
        float difference = AngleDifference(from, to);
        float absDifference = Abs(difference);
        return from + Clamp(delta, absDifference - Pi, absDifference) * (difference >= 0f ? 1f : -1f);
    }

    public static double RotateToward(double from, double to, double delta)
    {
        double difference = AngleDifference(from, to);
        double absDifference = Abs(difference);
        return from + Clamp(delta, absDifference - Math.PI, absDifference) * (difference >= 0.0 ? 1.0 : -1.0);
    }

    public static float SmoothStep(float from, float to, float weight)
    {
        if (IsEqualApprox(from, to))
        {
            return from;
        }

        float t = Clamp((weight - from) / (to - from), 0f, 1f);
        return t * t * (3f - 2f * t);
    }

    public static double SmoothStep(double from, double to, double weight)
    {
        if (IsEqualApprox(from, to))
        {
            return from;
        }

        double t = Clamp((weight - from) / (to - from), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }

    public static float Snapped(float s, float step)
    {
        return step != 0f ? Floor(s / step + 0.5f) * step : s;
    }

    public static double Snapped(double s, double step)
    {
        return step != 0.0 ? Floor(s / step + 0.5) * step : s;
    }

    public static int StepDecimals(double step)
    {
        double[] thresholds =
        {
            0.9999,
            0.09999,
            0.009999,
            0.0009999,
            9.999e-5,
            9.999e-6,
            9.999e-7,
            9.999e-8,
            9.999e-9
        };

        double fraction = Abs(step);
        fraction -= Math.Floor(fraction);

        for (int i = 0; i < thresholds.Length; i++)
        {
            if (fraction >= thresholds[i])
            {
                return i;
            }
        }

        return 0;
    }

    public static int Wrap(int value, int min, int max)
    {
        int range = max - min;
        return range == 0 ? min : min + ((value - min) % range + range) % range;
    }

    public static float Wrap(float value, float min, float max)
    {
        float range = max - min;
        return IsZeroApprox(range) ? min : min + ((value - min) % range + range) % range;
    }

    public static double Wrap(double value, double min, double max)
    {
        double range = max - min;
        return IsZeroApprox(range) ? min : min + ((value - min) % range + range) % range;
    }

    public static float PingPong(float value, float length)
    {
        static float Fract(float x) => x - Floor(x);

        if (length == 0f)
        {
            return 0f;
        }

        return Abs(Fract((value - length) / (length * 2f)) * length * 2f - length);
    }

    public static double PingPong(double value, double length)
    {
        static double Fract(double x) => x - Floor(x);

        if (length == 0.0)
        {
            return 0.0;
        }

        return Abs(Fract((value - length) / (length * 2.0)) * length * 2.0 - length);
    }

    public static int DecimalCount(double s) => DecimalCount((decimal)s);

    public static int DecimalCount(decimal s) => BitConverter.GetBytes(decimal.GetBits(s)[3])[2];

    // ---- integer rounders ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CeilToInt(float s) => (int)MathF.Ceiling(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CeilToInt(double s) => (int)Math.Ceiling(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FloorToInt(float s) => (int)MathF.Floor(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FloorToInt(double s) => (int)Math.Floor(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int RoundToInt(float s) => (int)MathF.Round(s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int RoundToInt(double s) => (int)Math.Round(s);
}
