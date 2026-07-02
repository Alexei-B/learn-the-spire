using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using Terminal.Gui;
using Attribute = Terminal.Gui.Attribute;
using Color = Terminal.Gui.Color;

namespace Lts2.Tui;

/// <summary>
/// The decision area: a focusable, scrollable list of options, each shown as a numbered header line
/// plus its (localized) description indented beneath it. Navigated with ↑/↓ + Enter, or the number
/// keys 1–9 as direct shortcuts. Raises <see cref="Activated"/> with the chosen option index.
/// </summary>
internal sealed class OptionsView : View
{
    /// <summary>One option: a coloured header label and its coloured (wrappable) description segments.</summary>
    public sealed record Entry(IReadOnlyList<Seg> Label, IReadOnlyList<Seg> Desc);

    private List<Entry> _entries = new();
    private List<string> _hotkeys = new();
    private int _selected = -1;
    private int _scroll;
    private int? _endTurn;

    // The option the Tab auto-play shortcut would apply (the default-strategy pick), or null when there
    // is none. Marked "(tab)" in the list and activated by Tab, just like a number-key shortcut.
    private int? _auto;

    public event Action<int>? Activated;

    /// <summary>Raised when the highlighted option changes (arrow keys, number keys, or SetEntries).</summary>
    public event Action<int>? SelectionChanged;

    public OptionsView()
    {
        CanFocus = true;
    }

    public int Selected => _selected;

    /// <summary>
    /// Set the options. <paramref name="endTurnIndex"/>, when given, is the option that always binds to
    /// the <c>0</c> key (End turn); the remaining options take the digits 1–9 in order.
    /// <paramref name="autoIndex"/>, when given, is the option the Tab shortcut applies (marked "(tab)").
    /// </summary>
    public void SetEntries(List<Entry> entries, int? endTurnIndex = null, int selected = 0, int? autoIndex = null)
    {
        _entries = entries;
        _endTurn = endTurnIndex;
        _auto = autoIndex is >= 0 && autoIndex < entries.Count ? autoIndex : null;
        _hotkeys = AssignHotkeys(entries.Count, endTurnIndex);
        _selected = entries.Count == 0 ? -1 : Math.Clamp(selected, 0, entries.Count - 1);
        _scroll = 0;
        SetNeedsDraw();
        SelectionChanged?.Invoke(_selected);
    }

    // Assign each option a digit shortcut: the end-turn option gets "0"; the rest take 1..9 in order
    // (anything past the ninth non-end-turn option gets no digit).
    private static List<string> AssignHotkeys(int count, int? endTurnIndex)
    {
        var keys = new List<string>(count);
        int next = 1;
        for (int i = 0; i < count; i++)
        {
            if (i == endTurnIndex)
            {
                keys.Add("0");
            }
            else
            {
                keys.Add(next <= 9 ? next.ToString() : "");
                next++;
            }
        }
        return keys;
    }

    private void Move(int delta)
    {
        if (_entries.Count == 0)
        {
            return;
        }
        _selected = Math.Clamp(_selected + delta, 0, _entries.Count - 1);
        SetNeedsDraw();
        SelectionChanged?.Invoke(_selected);
    }

    protected override bool OnKeyDown(Key key)
    {
        // Tab applies the default-strategy pick (marked "(tab)"), exactly like a number-key shortcut but
        // with a dynamically-chosen target. Handled before the empty-list guard and returned true so it
        // never falls through to Terminal.Gui's focus-navigation binding.
        if (key.KeyCode == KeyCode.Tab)
        {
            if (_auto is int a)
            {
                _selected = a;
                SetNeedsDraw();
                SelectionChanged?.Invoke(a);
                Activated?.Invoke(a);
            }
            return true;
        }
        if (_entries.Count == 0)
        {
            return false;
        }
        switch (key.KeyCode)
        {
            case KeyCode.CursorUp:
                Move(-1);
                return true;
            case KeyCode.CursorDown:
                Move(1);
                return true;
            case KeyCode.PageUp:
                Move(-5);
                return true;
            case KeyCode.PageDown:
                Move(5);
                return true;
            case KeyCode.Enter:
                if (_selected >= 0)
                {
                    Activated?.Invoke(_selected);
                }
                return true;
        }

        int ch = key.AsRune.Value;
        if (ch >= '0' && ch <= '9')
        {
            string want = ((char)ch).ToString();
            int n = _hotkeys.IndexOf(want);
            if (n >= 0)
            {
                _selected = n;
                SetNeedsDraw();
                SelectionChanged?.Invoke(n);
                Activated?.Invoke(n);
            }
            return true;
        }
        return false;
    }

    // Flatten the entries into render rows, recording where each option starts (for selection
    // highlight and scroll). Rebuilt each draw — cheap for a few dozen options.
    private (List<(List<Seg> segs, int opt, bool header)> rows, List<int> starts) Build(int width)
    {
        var rows = new List<(List<Seg>, int, bool)>();
        var starts = new List<int>();
        for (int i = 0; i < _entries.Count; i++)
        {
            starts.Add(rows.Count);
            Entry e = _entries[i];
            string key = i < _hotkeys.Count ? _hotkeys[i] : "";
            string tag = key.Length > 0 ? $"[{key}] " : "    ";
            var header = new List<Seg> { new(tag, Theme.Gold) };
            // Flag the default-strategy pick with a grey "(tab)" before the option, so the shortcut's
            // dynamic target is visible in the list.
            if (i == _auto)
            {
                header.Add(new Seg("(tab) ", Theme.Dim));
            }
            header.AddRange(e.Label);
            rows.Add((header, i, true));

            foreach (List<Seg> wrapped in Markup.Wrap(e.Desc, Math.Max(10, width - 6)))
            {
                var row = new List<Seg> { new("      ", Theme.Dim) };
                row.AddRange(wrapped);
                rows.Add((row, i, false));
            }
        }
        return (rows, starts);
    }

    protected override bool OnDrawingContent()
    {
        Rectangle vp = Viewport;
        SetAttribute(new Attribute(Theme.Fg, Theme.Bg));
        FillRect(vp, new Rune(' '));

        (List<(List<Seg> segs, int opt, bool header)> rows, List<int> starts) = Build(vp.Width);
        EnsureVisible(rows.Count, starts, vp.Height);

        var highlight = new Attribute(new Color(18, 20, 26), Theme.Teal);
        for (int r = 0; r < vp.Height; r++)
        {
            int idx = r + _scroll;
            if (idx < 0 || idx >= rows.Count)
            {
                continue;
            }
            (List<Seg> segs, int opt, bool header) = rows[idx];
            bool sel = opt == _selected;
            int col = 0;
            foreach (Seg seg in segs)
            {
                SetAttribute(sel && header ? highlight : new Attribute(seg.Fg, Theme.Bg));
                foreach (char c in seg.Text)
                {
                    if (col >= vp.Width)
                    {
                        break;
                    }
                    AddRune(col, r, new Rune(c));
                    col++;
                }
                if (col >= vp.Width)
                {
                    break;
                }
            }
            if (sel && header)
            {
                SetAttribute(highlight);
                for (; col < vp.Width; col++)
                {
                    AddRune(col, r, new Rune(' '));
                }
            }
        }
        return true;
    }

    private void EnsureVisible(int totalRows, List<int> starts, int height)
    {
        if (_selected < 0 || _selected >= starts.Count || height <= 0)
        {
            _scroll = 0;
            return;
        }
        int selStart = starts[_selected];
        int selEnd = (_selected + 1 < starts.Count ? starts[_selected + 1] : totalRows) - 1;
        if (selStart < _scroll)
        {
            _scroll = selStart;
        }
        else if (selEnd > _scroll + height - 1)
        {
            _scroll = Math.Max(selStart, selEnd - height + 1);
        }
        _scroll = Math.Max(0, Math.Min(_scroll, Math.Max(0, totalRows - height)));
    }

}
