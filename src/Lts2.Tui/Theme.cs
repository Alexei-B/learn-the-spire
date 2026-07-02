using Terminal.Gui;
using Attribute = Terminal.Gui.Attribute;

namespace Lts2.Tui;

/// <summary>
/// A modern, soft true-colour palette (One-Dark-ish) and the shared colour schemes, so the UI
/// reads like a contemporary terminal app rather than 16-colour DOS. Built once after
/// <c>Application.Init()</c>.
/// </summary>
internal static class Theme
{
    public static readonly Color Bg = new(24, 26, 33);
    public static readonly Color BgAlt = new(33, 36, 45);
    public static readonly Color Fg = new(208, 212, 222);
    public static readonly Color Dim = new(120, 126, 142);
    public static readonly Color Gold = new(221, 184, 97);
    public static readonly Color Teal = new(86, 182, 194);
    public static readonly Color Green = new(140, 200, 130);
    public static readonly Color Yellow = new(229, 192, 123);
    public static readonly Color Red = new(224, 108, 117);
    public static readonly Color Blue = new(97, 175, 239);
    public static readonly Color Magenta = new(198, 120, 221);
    public static readonly Color HpLost = new(64, 66, 78);   // already-lost HP on a health bar
    public static readonly Color LightGrey = new(176, 180, 192);
    public static readonly Color Orange = new(209, 154, 102);
    public static readonly Color Pink = new(224, 148, 208);
    public static readonly Color White = new(236, 238, 244);           // upgraded-card names read as brighter white/green

    // Card colour-identity borders for non-character cards (character cards reuse Red/Blue/Green/Magenta/Orange).
    public static readonly Color Colorless = new(130, 150, 175);       // colourless pool — a cool grey-blue
    public static readonly Color StatusTan = new(176, 160, 128);       // status cards — a muted grey-tan
    public static readonly Color Curse = new(140, 104, 168);           // curses — a dark purple

    // Rarity name-band backgrounds (the card name sits on this): dark enough for white/green text to pop.
    public static readonly Color BandCommon = new(58, 62, 72);
    public static readonly Color BandUncommon = new(38, 66, 100);
    public static readonly Color BandRare = new(102, 78, 40);

    /// <summary>HP colour: green when healthy, yellow when hurt, red when low.</summary>
    public static Color Hp(int cur, int max)
    {
        if (max <= 0)
        {
            return Dim;
        }
        double f = (double)cur / max;
        return f <= 0.3 ? Red : f <= 0.6 ? Yellow : Green;
    }

    public static ColorScheme Base { get; private set; } = null!;
    public static ColorScheme Frame { get; private set; } = null!;
    public static ColorScheme List { get; private set; } = null!;
    public static ColorScheme Menu { get; private set; } = null!;

    public static void Init()
    {
        // ColorScheme(normal, focus, hotNormal, hotFocus, disabled)
        Base = new ColorScheme(
            normal: new Attribute(Fg, Bg),
            focus: new Attribute(Bg, Teal),
            hotNormal: new Attribute(Gold, Bg),
            hotFocus: new Attribute(Bg, Teal),
            disabled: new Attribute(Dim, Bg));

        Frame = new ColorScheme(
            normal: new Attribute(Fg, Bg),
            focus: new Attribute(Teal, Bg),
            hotNormal: new Attribute(Gold, Bg),
            hotFocus: new Attribute(Gold, Bg),
            disabled: new Attribute(Dim, Bg));

        // The action list: clear, bright selection highlight when focused.
        List = new ColorScheme(
            normal: new Attribute(Fg, Bg),
            focus: new Attribute(new Color(18, 20, 26), Teal),
            hotNormal: new Attribute(Gold, Bg),
            hotFocus: new Attribute(new Color(18, 20, 26), Teal),
            disabled: new Attribute(Dim, Bg));

        Menu = new ColorScheme(
            normal: new Attribute(Fg, BgAlt),
            focus: new Attribute(Bg, Gold),
            hotNormal: new Attribute(Gold, BgAlt),
            hotFocus: new Attribute(Bg, Gold),
            disabled: new Attribute(Dim, BgAlt));
    }
}
