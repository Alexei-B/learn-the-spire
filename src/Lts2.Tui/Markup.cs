using System;
using System.Collections.Generic;
using System.Text;
using Color = Terminal.Gui.Color;

namespace Lts2.Tui;

/// <summary>
/// Converts the game's BBCode-ish localization markup into coloured <see cref="Seg"/>s, and renders
/// energy as teal circles. The game marks up loc strings with colour tags (<c>[gold]…[/gold]</c>,
/// <c>[blue]</c>, <c>[red]</c>, …), typography/effect tags we ignore (<c>[b]</c>, <c>[sine]</c>, …),
/// and energy icons as <c>[img]…_energy_icon.png[/img]</c>. Energy is shown as ● (filled) circles.
/// </summary>
internal static class Markup
{
    public const char Filled = '●';
    public const char Hollow = '○';

    /// <summary>Parse localization markup into coloured segments (default colour for untagged text).</summary>
    public static List<Seg> Parse(string? text, Color def)
    {
        var segs = new List<Seg>();
        if (string.IsNullOrEmpty(text))
        {
            return segs;
        }
        var stack = new Stack<Color>();
        Color cur = def;
        var buf = new StringBuilder();

        void Flush()
        {
            if (buf.Length > 0)
            {
                segs.Add(new Seg(buf.ToString(), cur));
                buf.Clear();
            }
        }

        int i = 0;
        while (i < text.Length)
        {
            if (text[i] != '[')
            {
                buf.Append(text[i]);
                i++;
                continue;
            }
            int end = text.IndexOf(']', i);
            if (end < 0)
            {
                buf.Append(text[i]);
                i++;
                continue;
            }
            string tag = text.Substring(i + 1, end - i - 1);

            if (tag.StartsWith("img", StringComparison.OrdinalIgnoreCase))
            {
                int close = text.IndexOf("[/img]", end + 1, StringComparison.OrdinalIgnoreCase);
                string inner = close > 0 ? text.Substring(end + 1, close - end - 1) : string.Empty;
                Flush();
                if (inner.Contains("energy_icon"))
                {
                    segs.Add(new Seg(Filled.ToString(), Theme.Teal));
                }
                else if (inner.Contains("star_icon"))
                {
                    segs.Add(new Seg("★", Theme.Gold));
                }
                // other images are dropped
                i = close > 0 ? close + "[/img]".Length : end + 1;
                continue;
            }

            if (tag.StartsWith("/", StringComparison.Ordinal))
            {
                if (ColorOf(tag.Substring(1)) is not null)
                {
                    Flush();
                    cur = stack.Count > 0 ? stack.Pop() : def;
                }
                i = end + 1;
                continue;
            }

            string name = tag.Split('=')[0];
            if (ColorOf(name) is { } c)
            {
                Flush();
                stack.Push(cur);
                cur = c;
            }
            // non-colour opening tags (effects/typography) are ignored, their text kept
            i = end + 1;
        }
        Flush();
        return segs;
    }

    private static Color? ColorOf(string name) => name.ToLowerInvariant() switch
    {
        "gold" => Theme.Gold,
        "blue" => Theme.Blue,
        "red" => Theme.Red,
        "purple" => Theme.Magenta,
        "green" => Theme.Green,
        "orange" => Theme.Orange,
        "aqua" => Theme.Teal,
        "pink" => Theme.Pink,
        _ => null,
    };

    /// <summary>Player energy as teal circles: ● per current energy, ○ up to max.</summary>
    public static List<Seg> EnergyCircles(int current, int max)
    {
        int filled = Math.Max(0, current);
        int hollow = Math.Max(0, max - current);
        var segs = new List<Seg>();
        if (filled > 0)
        {
            segs.Add(new Seg(new string(Filled, filled), Theme.Teal));
        }
        if (hollow > 0)
        {
            segs.Add(new Seg(new string(Hollow, hollow), Theme.Teal));
        }
        if (filled == 0 && hollow == 0)
        {
            segs.Add(new Seg(Hollow.ToString(), Theme.Teal));
        }
        return segs;
    }

    /// <summary>A card's energy cost as teal circles (● per energy), or X / a number when large.</summary>
    public static Seg Cost(bool costsX, int cost)
    {
        if (costsX)
        {
            return new Seg("X", Theme.Teal);
        }
        if (cost <= 0)
        {
            return new Seg(Hollow.ToString(), Theme.Teal);
        }
        return cost <= 5
            ? new Seg(new string(Filled, cost), Theme.Teal)
            : new Seg($"{cost}{Filled}", Theme.Teal);
    }

    /// <summary>Greedy word-wrap over coloured segments; splits on spaces and explicit newlines.</summary>
    public static List<List<Seg>> Wrap(IReadOnlyList<Seg> segs, int width)
    {
        var lines = new List<List<Seg>>();
        var line = new List<Seg>();
        int len = 0;

        void NewLine()
        {
            lines.Add(line);
            line = new List<Seg>();
            len = 0;
        }

        foreach (Seg seg in segs)
        {
            int start = 0;
            for (int k = 0; k <= seg.Text.Length; k++)
            {
                bool boundary = k == seg.Text.Length || seg.Text[k] == ' ' || seg.Text[k] == '\n';
                if (!boundary)
                {
                    continue;
                }
                if (k > start)
                {
                    string word = seg.Text.Substring(start, k - start);
                    if (len > 0 && len + 1 + word.Length > width)
                    {
                        NewLine();
                    }
                    if (len > 0)
                    {
                        line.Add(new Seg(" ", Theme.Fg));
                        len++;
                    }
                    line.Add(new Seg(word, seg.Fg));
                    len += word.Length;
                }
                if (k < seg.Text.Length && seg.Text[k] == '\n')
                {
                    NewLine();
                }
                start = k + 1;
            }
        }
        if (line.Count > 0)
        {
            lines.Add(line);
        }
        return lines;
    }
}
