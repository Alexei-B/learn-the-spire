using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using Terminal.Gui;
using Attribute = Terminal.Gui.Attribute;
using Color = Terminal.Gui.Color;

namespace Lts2.Tui;

/// <summary>One coloured run of text within a board line.</summary>
internal readonly record struct Seg(string Text, Color Fg);

/// <summary>A single board line: an ordered list of coloured segments. Fluent to build.</summary>
internal sealed class Line : List<Seg>
{
    public Line Add(string text, Color fg)
    {
        base.Add(new Seg(text, fg));
        return this;
    }

    public Line T(string text) => Add(text, Theme.Fg);
    public Line Dim(string text) => Add(text, Theme.Dim);
}

/// <summary>
/// A view that paints a list of coloured <see cref="Line"/>s — the "canvas" for the game board.
/// Terminal.Gui has no rich-text markup, so we draw segment by segment with per-run attributes.
/// </summary>
internal sealed class BoardView : View
{
    private IReadOnlyList<Line> _lines = new List<Line>();
    private Func<int, int, IReadOnlyList<Line>>? _render;

    public BoardView()
    {
        CanFocus = false;
    }

    /// <summary>Set static lines (used by popups). Clears any width-aware renderer.</summary>
    public void SetLines(IReadOnlyList<Line> lines)
    {
        _render = null;
        _lines = lines;
        SetNeedsDraw();
    }

    /// <summary>
    /// Set a width-aware renderer, invoked with the current content width each draw — so content that
    /// depends on width (health bars, two-column combat, the map) reflows on resize automatically.
    /// </summary>
    public void SetRenderer(Func<int, int, IReadOnlyList<Line>> render)
    {
        _render = render;
        SetNeedsDraw();
    }

    protected override bool OnDrawingContent()
    {
        Rectangle vp = Viewport;
        SetAttribute(new Attribute(Theme.Fg, Theme.Bg));
        FillRect(vp, new Rune(' '));

        IReadOnlyList<Line> lines = _render is not null ? _render(vp.Width, vp.Height) : _lines;
        for (int row = 0; row < lines.Count && row < vp.Height; row++)
        {
            int col = 0;
            foreach (Seg seg in lines[row])
            {
                SetAttribute(new Attribute(seg.Fg, Theme.Bg));
                foreach (char ch in seg.Text)
                {
                    if (col >= vp.Width)
                    {
                        break;
                    }
                    AddRune(col, row, new Rune(ch));
                    col++;
                }
                if (col >= vp.Width)
                {
                    break;
                }
            }
        }
        return true;
    }
}
