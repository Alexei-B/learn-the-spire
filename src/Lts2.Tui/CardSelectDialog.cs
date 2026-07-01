using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Lts2.Harness;
using Terminal.Gui;
using Attribute = Terminal.Gui.Attribute;
using Color = Terminal.Gui.Color;

namespace Lts2.Tui;

/// <summary>
/// A modal picker for a mid-effect multi-card choice (<see cref="GamePhase.Choice"/> with
/// <see cref="PendingChoiceView.MaxSelect"/> &gt; 1) — e.g. the Regent's CHARGE!! picking cards from
/// the draw pile, or a Neow reward that selects several deck cards. The player toggles cards on/off
/// (Space or the card's number) and confirms (Enter/Confirm) once the count is within the choice's
/// [min, max]. Returns the chosen indices into <see cref="PendingChoiceView.Options"/>, or null if
/// the picker was dismissed without confirming (the choice stays pending, to be re-opened).
/// </summary>
internal static class CardSelectDialog
{
    public static IReadOnlyList<int>? Show(PendingChoiceView choice)
    {
        int n = choice.Options.Count;
        int min = Math.Clamp(choice.MinSelect, 0, n);
        int max = Math.Clamp(choice.MaxSelect, min, n);

        var list = new CardPickList(choice.Options, min, max)
        {
            X = 0,
            Y = 0,
            Width = Dim.Fill(),
            Height = Dim.Fill(2),
        };

        var status = new Label { X = 1, Y = Pos.AnchorEnd(1), Width = Dim.Fill(1), Text = "" };

        string verb = choice.IsUpgradeSelection ? "forge" : "select";
        string range = min == max ? $"{max}" : $"{min}-{max}";
        var dlg = new Dialog
        {
            Title = $"Choose {range} card(s) to {verb}",
            Width = Math.Min(78, Math.Max(46, LongestLine(choice.Options) + 10)),
            Height = Math.Min(24, n + 7),
            ColorScheme = Theme.Base,
        };

        IReadOnlyList<int>? result = null;

        void UpdateStatus()
        {
            int count = list.CheckedCount;
            bool ok = count >= min && count <= max;
            string hint = min == 0 && count == 0 ? "  (Enter to skip)" : ok ? "  (Enter to confirm)" : "";
            status.Text = $" Selected {count}/{max}{hint}   Space toggle · ↑↓ move · Esc cancel";
        }

        void Confirm()
        {
            int count = list.CheckedCount;
            if (count < min || count > max)
            {
                return; // not a valid selection yet; ignore
            }
            result = list.CheckedIndices;
            Application.RequestStop(dlg);
        }

        list.Changed += UpdateStatus;
        list.ConfirmRequested += Confirm;

        var confirm = new Button { Text = "Confirm", IsDefault = true };
        confirm.Accepting += (_, e) => { e.Cancel = true; Confirm(); };

        var cancel = new Button { Text = "Cancel" };
        cancel.Accepting += (_, e) => { e.Cancel = true; result = null; Application.RequestStop(dlg); };

        dlg.Add(list, status);
        dlg.AddButton(confirm);
        dlg.AddButton(cancel);

        UpdateStatus();
        list.SetFocus();
        Application.Run(dlg);
        dlg.Dispose();
        return result;
    }

    private static int LongestLine(IReadOnlyList<CardView> cards) =>
        cards.Count == 0 ? 0 : cards.Max(c => BoardRenderer.CardDisplayName(c).Length + 8);
}

/// <summary>
/// The scrollable, toggleable card list inside <see cref="CardSelectDialog"/>. Each card is a header
/// line (a checkbox plus the card's coloured label) with its rules text wrapped beneath. ↑/↓ move the
/// highlight, Space or the card's digit toggles it, Enter requests confirmation.
/// </summary>
internal sealed class CardPickList : View
{
    private readonly IReadOnlyList<CardView> _cards;
    private readonly bool[] _checked;
    private readonly int _min;
    private readonly int _max;
    private int _selected;
    private int _scroll;

    /// <summary>Raised whenever the highlight moves or a card is toggled (so the dialog can restatus).</summary>
    public event Action? Changed;

    /// <summary>Raised when the player presses Enter to confirm the current selection.</summary>
    public event Action? ConfirmRequested;

    public CardPickList(IReadOnlyList<CardView> cards, int min, int max)
    {
        _cards = cards;
        _checked = new bool[cards.Count];
        _min = min;
        _max = max;
        _selected = cards.Count == 0 ? -1 : 0;
        CanFocus = true;
    }

    public int CheckedCount => _checked.Count(b => b);

    public IReadOnlyList<int> CheckedIndices =>
        Enumerable.Range(0, _checked.Length).Where(i => _checked[i]).ToList();

    private void Move(int delta)
    {
        if (_cards.Count == 0)
        {
            return;
        }
        _selected = Math.Clamp(_selected + delta, 0, _cards.Count - 1);
        SetNeedsDraw();
        Changed?.Invoke();
    }

    // Toggle a card, honouring the max: a checked card always unchecks; an unchecked one only checks
    // when there is room. A max of 1 behaves like a radio — checking one clears the previous pick.
    private void Toggle(int i)
    {
        if (i < 0 || i >= _checked.Length)
        {
            return;
        }
        if (_checked[i])
        {
            _checked[i] = false;
        }
        else if (_max == 1)
        {
            System.Array.Clear(_checked, 0, _checked.Length);
            _checked[i] = true;
        }
        else if (CheckedCount < _max)
        {
            _checked[i] = true;
        }
        SetNeedsDraw();
        Changed?.Invoke();
    }

    protected override bool OnKeyDown(Key key)
    {
        if (_cards.Count == 0)
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
            case KeyCode.Space:
                Toggle(_selected);
                return true;
            case KeyCode.Enter:
                ConfirmRequested?.Invoke();
                return true;
        }

        int ch = key.AsRune.Value;
        if (ch >= '1' && ch <= '9')
        {
            int idx = ch - '1';
            if (idx < _cards.Count)
            {
                _selected = idx;
                Toggle(idx);
            }
            return true;
        }
        return false;
    }

    // Flatten cards into render rows (header + wrapped description), recording each card's start row
    // so the highlight and scroll can track it. Rebuilt each draw — cheap for a handful of cards.
    private (List<(List<Seg> segs, int card, bool header)> rows, List<int> starts) Build(int width)
    {
        var rows = new List<(List<Seg>, int, bool)>();
        var starts = new List<int>();
        for (int i = 0; i < _cards.Count; i++)
        {
            starts.Add(rows.Count);
            CardView c = _cards[i];
            string digit = i < 9 ? $"{i + 1}" : " ";
            var header = new List<Seg>
            {
                new($"[{digit}] ", Theme.Gold),
                new(_checked[i] ? "[x] " : "[ ] ", _checked[i] ? Theme.Green : Theme.Dim),
            };
            header.AddRange(BoardRenderer.CardSegs(c));
            rows.Add((header, i, true));

            var desc = Markup.Parse(BoardRenderer.CardDescription(c), Theme.Fg);
            foreach (List<Seg> wrapped in Markup.Wrap(desc, Math.Max(10, width - 8)))
            {
                var row = new List<Seg> { new("        ", Theme.Dim) };
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

        (List<(List<Seg> segs, int card, bool header)> rows, List<int> starts) = Build(vp.Width);
        EnsureVisible(rows.Count, starts, vp.Height);

        var highlight = new Attribute(new Color(18, 20, 26), Theme.Teal);
        for (int r = 0; r < vp.Height; r++)
        {
            int idx = r + _scroll;
            if (idx < 0 || idx >= rows.Count)
            {
                continue;
            }
            (List<Seg> segs, int card, bool header) = rows[idx];
            bool sel = card == _selected;
            int col = 0;
            foreach (Seg seg in segs)
            {
                SetAttribute(sel && header ? highlight : new Attribute(seg.Fg, Theme.Bg));
                foreach (char ccar in seg.Text)
                {
                    if (col >= vp.Width)
                    {
                        break;
                    }
                    AddRune(col, r, new Rune(ccar));
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
