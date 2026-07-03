using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Lts2.Harness;
using Lts2.Localization;
using Terminal.Gui;
using Attribute = Terminal.Gui.Attribute;
using Color = Terminal.Gui.Color;

namespace Lts2.Tui;

/// <summary>
/// The combat decision area, drawn as the actual hand of cards rather than a list. Cards are laid out
/// left-to-right (wrapping to further rows) each with a hotkey indicator above it; an End Turn button
/// (hotkey <c>0</c>) sits at the far right with the potion belt drawn right-to-left above it.
///
/// Playing a card: press its hotkey (or click it). An untargeted card resolves immediately; a targeted
/// card enters <em>targeting mode</em> — the enemies gain hotkey badges (with the projected damage) via
/// <see cref="TargetOverlay"/> and pressing a target hotkey (or Esc to cancel) resolves it. Targeted
/// cards always go through targeting, even with a single legal target, so the predicted effect is shown.
///
/// The view is purely a controller/renderer over the harness options: it raises <see cref="Chosen"/>
/// with the <see cref="GameOption"/> to apply, and <see cref="TargetingChanged"/> whenever the board's
/// target overlay should be repainted.
/// </summary>
internal sealed class CombatDecisionView : View
{
    // Layout metrics (columns/rows). The card size comes from the shared card-art renderer.
    private const int CardGap = 1;        // blank columns between adjacent cards
    private const int RowGap = 1;         // blank rows between card rows
    private const int HotkeyRow = 1;      // the "[n]" indicator line above each card
    private const int RegionGap = 2;      // columns between the cards region and the right-hand region
    private const int EndTurnW = 11;      // End Turn button width
    private const int EndTurnH = 4;       // End Turn button height (box lines)
    private const int PotionW = 12;       // potion slot width
    private const int PotionH = 3;        // potion slot height

    private static int CardW => BoardRenderer.CardArtWidth;
    private static int CardBlockH => HotkeyRow + BoardRenderer.CardArtHeight;

    /// <summary>One hand card and the play option(s) that resolve it.</summary>
    private sealed class Slot
    {
        public CardView Card = null!;
        public List<GameOption> Options = new();
        public bool Playable;
        public bool Targeted;
        public string? Hotkey;     // "1".."9", or null (unplayable / past the ninth)
        public bool IsAuto;        // the default-strategy ("tab") pick
    }

    private GameState? _state;
    private List<Slot> _slots = new();
    private List<string?> _potions = new();
    private GameOption? _endTurn;
    private GameOption? _auto;

    // The arrow-key selection: an index into _slots while choosing a card, into _targetOptions while aiming.
    // Enter activates whatever is selected.
    private int _selCard;
    private int _selTarget;

    // Targeting mode: the card being aimed and its per-target options in hotkey order.
    private Slot? _targeting;
    private List<GameOption> _targetOptions = new();

    /// <summary>Raised with the option to apply (a card play, a resolved target, or End Turn).</summary>
    public event Action<GameOption>? Chosen;

    /// <summary>Raised when the board's target overlay should be repainted (targeting started/cancelled).</summary>
    public event Action? TargetingChanged;

    public CombatDecisionView()
    {
        CanFocus = true;
    }

    /// <summary>The hotkey badges to overlay on the enemies while targeting, or null when not targeting.</summary>
    public IReadOnlyDictionary<uint, TargetBadge>? TargetOverlay { get; private set; }

    /// <summary>A one-line status hint for the current mode (targeting prompt), or null.</summary>
    public string? Prompt { get; private set; }

    /// <summary>
    /// Rebuild the hand/options model from a fresh combat state (resets any targeting).
    /// <paramref name="recommended"/> is the decision engine's suggested move (the Tab "auto-play" pick),
    /// or null if it has none — computed by the caller so the strategy engine is swappable.
    /// </summary>
    public void SetState(GameState state, IReadOnlyList<GameOption> options, GameOption? recommended)
    {
        _state = state;
        _endTurn = options.FirstOrDefault(o => o.Kind == OptionKind.EndTurn);
        _potions = state.Players.Count > 0 ? state.Players[0].Potions.ToList() : new List<string?>();
        _auto = recommended;

        Dictionary<int, List<GameOption>> byIndex = options
            .Where(o => o.Kind == OptionKind.PlayCard && o.HandIndex is not null)
            .GroupBy(o => o.HandIndex!.Value)
            .ToDictionary(g => g.Key, g => g.ToList());

        IReadOnlyList<CardView> hand = state.Players.Count > 0 && state.Players[0].CombatState is { } cs
            ? cs.Hand
            : Array.Empty<CardView>();

        _slots = new List<Slot>(hand.Count);
        int nextKey = 1;
        for (int i = 0; i < hand.Count; i++)
        {
            byIndex.TryGetValue(i, out List<GameOption>? opts);
            bool playable = opts is { Count: > 0 };
            var slot = new Slot
            {
                Card = hand[i],
                Options = opts ?? new List<GameOption>(),
                Playable = playable,
                Targeted = playable && opts!.Any(o => o.TargetCombatId is not null),
                IsAuto = _auto?.Kind == OptionKind.PlayCard && _auto.HandIndex == i,
            };
            if (playable && nextKey <= 9)
            {
                slot.Hotkey = nextKey.ToString();
                nextKey++;
            }
            _slots.Add(slot);
        }

        _selCard = 0;
        ClearTargeting();
        SetNeedsDraw();
    }

    // ---- Layout ----------------------------------------------------------------

    private static int FilledPotionCount(IReadOnlyList<string?> potions) => potions.Count(p => p is not null);

    private static int RightRegionWidth(int potionCount)
    {
        int potionsW = potionCount > 0 ? potionCount * PotionW + (potionCount - 1) : 0;
        return Math.Max(EndTurnW, potionsW);
    }

    private (int perRow, int rightW) Layout(int width)
    {
        int rightW = RightRegionWidth(FilledPotionCount(_potions));
        int leftW = Math.Max(CardW, width - rightW - RegionGap);
        int perRow = Math.Max(1, (leftW + CardGap) / (CardW + CardGap));
        return (perRow, rightW);
    }

    /// <summary>
    /// The content height (rows) the decision area needs to show a hand of <paramref name="handCount"/>
    /// cards and <paramref name="potionCount"/> potions within <paramref name="innerWidth"/> columns —
    /// enough for however many card rows the hand wraps into (and at least the right-hand region's height).
    /// </summary>
    public static int ContentHeight(int handCount, int potionCount, int innerWidth)
    {
        int rightW = RightRegionWidth(potionCount);
        int leftW = Math.Max(CardW, innerWidth - rightW - RegionGap);
        int perRow = Math.Max(1, (leftW + CardGap) / (CardW + CardGap));
        int rows = Math.Max(1, (handCount + perRow - 1) / perRow);
        int cardsH = rows * CardBlockH + (rows - 1) * RowGap;
        int rightH = (potionCount > 0 ? PotionH + 1 : 0) + 1 /* [0] label */ + EndTurnH;
        return Math.Max(cardsH, rightH);
    }

    // ---- Input -----------------------------------------------------------------

    protected override bool OnKeyDown(Key key)
    {
        if (_state is null)
        {
            return false;
        }

        // Esc / Backspace cancel aiming and return to card selection.
        if (key.KeyCode is KeyCode.Esc or KeyCode.Backspace)
        {
            if (_targeting is not null)
            {
                CancelTargeting();
                return true;
            }
            return false;
        }

        if (_targeting is not null)
        {
            switch (key.KeyCode)
            {
                case KeyCode.CursorLeft:
                case KeyCode.CursorUp:
                    MoveTarget(-1);
                    return true;
                case KeyCode.CursorRight:
                case KeyCode.CursorDown:
                    MoveTarget(1);
                    return true;
                case KeyCode.Enter:
                    if (_selTarget >= 0 && _selTarget < _targetOptions.Count)
                    {
                        Choose(_targetOptions[_selTarget]);
                    }
                    return true;
            }
            if (TryDigit(key, out int t) && t >= 1 && t <= _targetOptions.Count)
            {
                Choose(_targetOptions[t - 1]);
            }
            return true; // swallow everything else while aiming
        }

        // Card selection mode.
        switch (key.KeyCode)
        {
            case KeyCode.CursorLeft:
                MoveCard(-1);
                return true;
            case KeyCode.CursorRight:
                MoveCard(1);
                return true;
            case KeyCode.CursorUp:
                MoveCard(-PerRow());
                return true;
            case KeyCode.CursorDown:
                MoveCard(PerRow());
                return true;
            case KeyCode.Enter:
                if (_selCard >= 0 && _selCard < _slots.Count)
                {
                    Activate(_slots[_selCard]);
                }
                return true;
            case KeyCode.Tab:
                ActivateAuto();
                return true;
        }
        if (TryDigit(key, out int d))
        {
            if (d == 0)
            {
                if (_endTurn is not null)
                {
                    Choose(_endTurn);
                }
                return true;
            }
            Slot? slot = _slots.FirstOrDefault(s => s.Hotkey == d.ToString());
            if (slot is not null)
            {
                _selCard = _slots.IndexOf(slot);
                Activate(slot);
            }
            return true;
        }
        return false;
    }

    private int PerRow() => Layout(Math.Max(1, Viewport.Width)).perRow;

    private void MoveCard(int delta)
    {
        if (_slots.Count == 0)
        {
            return;
        }
        _selCard = Math.Clamp(_selCard + delta, 0, _slots.Count - 1);
        SetNeedsDraw();
    }

    private void MoveTarget(int delta)
    {
        if (_targetOptions.Count == 0)
        {
            return;
        }
        _selTarget = Math.Clamp(_selTarget + delta, 0, _targetOptions.Count - 1);
        RebuildOverlay();
        TargetingChanged?.Invoke();
        SetNeedsDraw();
    }

    /// <summary>Resolve the aiming card against a specific enemy (a click on the combat board).</summary>
    public void SelectTargetByCombatId(uint combatId)
    {
        GameOption? option = _targetOptions.FirstOrDefault(o => o.TargetCombatId == combatId);
        if (option is not null)
        {
            Choose(option);
        }
    }

    /// <summary>Cancel aiming from an external source (e.g. a right-click on the board).</summary>
    public void StopTargeting()
    {
        if (_targeting is not null)
        {
            CancelTargeting();
        }
    }

    /// <summary>True while aiming a targeted card.</summary>
    public bool IsTargeting => _targeting is not null;

    protected override bool OnMouseEvent(MouseEventArgs mouseEvent)
    {
        if (_state is null)
        {
            return false;
        }
        bool right = mouseEvent.Flags.HasFlag(MouseFlags.Button3Clicked);
        bool left = mouseEvent.IsSingleClicked;
        if (!left && !right)
        {
            return false;
        }
        SetFocus();
        if (right)
        {
            // Right-click anywhere in the decision area cancels aiming.
            if (_targeting is not null)
            {
                CancelTargeting();
            }
            return true;
        }
        int mx = mouseEvent.Position.X, my = mouseEvent.Position.Y;
        if (HitEndTurn(mx, my))
        {
            if (_endTurn is not null)
            {
                Choose(_endTurn);
            }
            return true;
        }
        Slot? slot = HitCard(mx, my);
        if (slot is not null)
        {
            _selCard = _slots.IndexOf(slot);
            Activate(slot);
            return true;
        }
        return false;
    }

    // Tab plays the default-strategy pick outright — including the target it chose to attack — rather than
    // dropping into targeting, so a single keypress resolves the whole suggested move.
    private void ActivateAuto()
    {
        if (_auto is { } a)
        {
            Choose(a);
        }
    }

    private void Activate(Slot slot)
    {
        if (!slot.Playable)
        {
            return;
        }
        if (slot.Targeted)
        {
            BeginTargeting(slot);
        }
        else
        {
            Choose(slot.Options[0]);
        }
    }

    private void Choose(GameOption option)
    {
        ClearTargeting();
        Chosen?.Invoke(option);
    }

    private static bool TryDigit(Key key, out int digit)
    {
        int ch = key.AsRune.Value;
        if (ch >= '0' && ch <= '9')
        {
            digit = ch - '0';
            return true;
        }
        digit = -1;
        return false;
    }

    // ---- Targeting -------------------------------------------------------------

    private void BeginTargeting(Slot slot)
    {
        _targeting = slot;
        _targetOptions = slot.Options.Where(o => o.TargetCombatId is not null).ToList();
        _selTarget = 0;
        RebuildOverlay();
        TargetingChanged?.Invoke();
        SetNeedsDraw();
    }

    // (Re)build the enemy target overlay for the current aiming state, flagging the selected target.
    private void RebuildOverlay()
    {
        if (_targeting is null)
        {
            return;
        }
        var overlay = new Dictionary<uint, TargetBadge>();
        for (int k = 0; k < _targetOptions.Count; k++)
        {
            GameOption o = _targetOptions[k];
            overlay[o.TargetCombatId!.Value] = new TargetBadge((k + 1).ToString(), o.Card?.Damage, k == _selTarget);
        }
        TargetOverlay = overlay;
        Prompt = $"Aiming {BoardRenderer.CardDisplayName(_targeting.Card)} — ◂▸ or 1-{_targetOptions.Count} pick target · Enter/click hits · Esc cancels";
    }

    private void CancelTargeting()
    {
        bool notify = _targeting is not null;
        ClearTargeting();
        if (notify)
        {
            TargetingChanged?.Invoke();
        }
    }

    private void ClearTargeting()
    {
        _targeting = null;
        _targetOptions = new List<GameOption>();
        TargetOverlay = null;
        Prompt = null;
        SetNeedsDraw();
    }

    // ---- Drawing ---------------------------------------------------------------

    protected override bool OnDrawingContent()
    {
        Rectangle vp = Viewport;
        SetAttribute(new Attribute(Theme.Fg, Theme.Bg));
        FillRect(vp, new Rune(' '));
        if (_state is null)
        {
            return true;
        }

        (int perRow, int rightW) = Layout(vp.Width);

        for (int i = 0; i < _slots.Count; i++)
        {
            int row = i / perRow, col = i % perRow;
            int x = col * (CardW + CardGap);
            int y = row * (CardBlockH + RowGap);
            bool selected = _targeting is null && i == _selCard;
            DrawSlot(_slots[i], x, y, selected);
        }

        DrawRightRegion(vp.Width);
        return true;
    }

    private void DrawSlot(Slot slot, int x, int y, bool selected)
    {
        bool dim = !slot.Playable || (_targeting is not null && _targeting != slot);

        // Hotkey indicator above the card: "[tab]" for the recommended pick, "[n]" otherwise, "[aim]"
        // for the card currently being targeted. The arrow-key selection is drawn reversed (teal ground).
        string label = _targeting == slot ? "[aim]"
            : slot.IsAuto ? "[tab]"
            : slot.Hotkey is { } h ? $"[{h}]"
            : "";
        Color labelFg = _targeting == slot || slot.IsAuto ? Theme.Teal : Theme.Gold;
        if (selected)
        {
            if (label.Length == 0)
            {
                label = "[ ]";
            }
            DrawStr(x + 1, y, "▸" + label, Theme.Bg, Theme.Teal);
        }
        else if (label.Length > 0)
        {
            DrawStr(x + 1, y, label, labelFg);
        }

        List<Line> art = BoardRenderer.CardArt(slot.Card, dim);
        for (int r = 0; r < art.Count; r++)
        {
            DrawLineAt(x, y + HotkeyRow + r, art[r]);
        }
    }

    private void DrawRightRegion(int width)
    {
        int filled = FilledPotionCount(_potions);

        // Potion belt across the top of the right region, drawn right-to-left (slot 0 rightmost).
        int shown = 0;
        for (int i = 0; i < _potions.Count; i++)
        {
            if (_potions[i] is not { } potionId)
            {
                continue;
            }
            int px = width - PotionW - shown * (PotionW + 1);
            DrawPotion(px, 0, potionId);
            shown++;
        }

        // End Turn button below the potions, at the far right, with its "[0]" hotkey above it.
        int ey = filled > 0 ? PotionH + 1 : 0;
        int ex = width - EndTurnW;
        DrawStr(ex + (EndTurnW - 3) / 2, ey, "[0]", Theme.Gold);
        DrawBox(ex, ey + 1, EndTurnW, EndTurnH, Theme.Gold);
        DrawCentered(ex, ey + 2, EndTurnW, "End", Theme.Fg);
        DrawCentered(ex, ey + 3, EndTurnW, "Turn", Theme.Fg);
    }

    private void DrawPotion(int x, int y, string potionId)
    {
        DrawBox(x, y, PotionW, PotionH, Theme.Teal);
        string name = Localizer.PotionName(potionId);
        if (name.Length > PotionW - 2)
        {
            name = name.Substring(0, PotionW - 2);
        }
        DrawCentered(x, y + 1, PotionW, name, Theme.Teal);
    }

    // ---- Hit testing -----------------------------------------------------------

    private Slot? HitCard(int mx, int my)
    {
        (int perRow, _) = Layout(Viewport.Width);
        for (int i = 0; i < _slots.Count; i++)
        {
            int row = i / perRow, col = i % perRow;
            int x = col * (CardW + CardGap);
            int y = row * (CardBlockH + RowGap);
            if (mx >= x && mx < x + CardW && my >= y && my < y + CardBlockH)
            {
                return _slots[i];
            }
        }
        return null;
    }

    private bool HitEndTurn(int mx, int my)
    {
        int filled = FilledPotionCount(_potions);
        int ey = filled > 0 ? PotionH + 1 : 0;
        int ex = Viewport.Width - EndTurnW;
        return mx >= ex && mx < ex + EndTurnW && my >= ey && my < ey + 1 + EndTurnH;
    }

    // ---- Low-level cell drawing ------------------------------------------------

    private void Put(int x, int y, Rune rune, Color fg, Color? bg = null)
    {
        if (x < 0 || y < 0 || x >= Viewport.Width || y >= Viewport.Height)
        {
            return;
        }
        SetAttribute(new Attribute(fg, bg ?? Theme.Bg));
        AddRune(x, y, rune);
    }

    private void DrawStr(int x, int y, string s, Color fg, Color? bg = null)
    {
        for (int i = 0; i < s.Length; i++)
        {
            Put(x + i, y, new Rune(s[i]), fg, bg);
        }
    }

    private void DrawCentered(int x, int y, int width, string s, Color fg)
    {
        if (s.Length > width)
        {
            s = s.Substring(0, width);
        }
        DrawStr(x + (width - s.Length) / 2, y, s, fg);
    }

    private void DrawLineAt(int x, int y, Line line)
    {
        int col = x;
        foreach (Seg seg in line)
        {
            foreach (char ch in seg.Text)
            {
                Put(col, y, new Rune(ch), seg.Fg, seg.Bg);
                col++;
            }
        }
    }

    private void DrawBox(int x, int y, int w, int h, Color color)
    {
        DrawStr(x, y, "┌" + new string('─', w - 2) + "┐", color);
        for (int r = 1; r < h - 1; r++)
        {
            Put(x, y + r, new Rune('│'), color);
            Put(x + w - 1, y + r, new Rune('│'), color);
        }
        DrawStr(x, y + h - 1, "└" + new string('─', w - 2) + "┘", color);
    }
}
