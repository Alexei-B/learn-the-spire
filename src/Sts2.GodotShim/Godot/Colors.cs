using System.Collections.Frozen;
using System.Collections.Generic;

namespace Godot;

public static class Colors
{
	internal static readonly FrozenDictionary<string, Color> NamedColors = new Dictionary<string, Color>
	{
		{ "ALICEBLUE", AliceBlue },
		{ "ANTIQUEWHITE", AntiqueWhite },
		{ "AQUA", Aqua },
		{ "AQUAMARINE", Aquamarine },
		{ "AZURE", Azure },
		{ "BEIGE", Beige },
		{ "BISQUE", Bisque },
		{ "BLACK", Black },
		{ "BLANCHEDALMOND", BlanchedAlmond },
		{ "BLUE", Blue },
		{ "BLUEVIOLET", BlueViolet },
		{ "BROWN", Brown },
		{ "BURLYWOOD", Burlywood },
		{ "CADETBLUE", CadetBlue },
		{ "CHARTREUSE", Chartreuse },
		{ "CHOCOLATE", Chocolate },
		{ "CORAL", Coral },
		{ "CORNFLOWERBLUE", CornflowerBlue },
		{ "CORNSILK", Cornsilk },
		{ "CRIMSON", Crimson },
		{ "CYAN", Cyan },
		{ "DARKBLUE", DarkBlue },
		{ "DARKCYAN", DarkCyan },
		{ "DARKGOLDENROD", DarkGoldenrod },
		{ "DARKGRAY", DarkGray },
		{ "DARKGREEN", DarkGreen },
		{ "DARKKHAKI", DarkKhaki },
		{ "DARKMAGENTA", DarkMagenta },
		{ "DARKOLIVEGREEN", DarkOliveGreen },
		{ "DARKORANGE", DarkOrange },
		{ "DARKORCHID", DarkOrchid },
		{ "DARKRED", DarkRed },
		{ "DARKSALMON", DarkSalmon },
		{ "DARKSEAGREEN", DarkSeaGreen },
		{ "DARKSLATEBLUE", DarkSlateBlue },
		{ "DARKSLATEGRAY", DarkSlateGray },
		{ "DARKTURQUOISE", DarkTurquoise },
		{ "DARKVIOLET", DarkViolet },
		{ "DEEPPINK", DeepPink },
		{ "DEEPSKYBLUE", DeepSkyBlue },
		{ "DIMGRAY", DimGray },
		{ "DODGERBLUE", DodgerBlue },
		{ "FIREBRICK", Firebrick },
		{ "FLORALWHITE", FloralWhite },
		{ "FORESTGREEN", ForestGreen },
		{ "FUCHSIA", Fuchsia },
		{ "GAINSBORO", Gainsboro },
		{ "GHOSTWHITE", GhostWhite },
		{ "GOLD", Gold },
		{ "GOLDENROD", Goldenrod },
		{ "GRAY", Gray },
		{ "GREEN", Green },
		{ "GREENYELLOW", GreenYellow },
		{ "HONEYDEW", Honeydew },
		{ "HOTPINK", HotPink },
		{ "INDIANRED", IndianRed },
		{ "INDIGO", Indigo },
		{ "IVORY", Ivory },
		{ "KHAKI", Khaki },
		{ "LAVENDER", Lavender },
		{ "LAVENDERBLUSH", LavenderBlush },
		{ "LAWNGREEN", LawnGreen },
		{ "LEMONCHIFFON", LemonChiffon },
		{ "LIGHTBLUE", LightBlue },
		{ "LIGHTCORAL", LightCoral },
		{ "LIGHTCYAN", LightCyan },
		{ "LIGHTGOLDENROD", LightGoldenrod },
		{ "LIGHTGRAY", LightGray },
		{ "LIGHTGREEN", LightGreen },
		{ "LIGHTPINK", LightPink },
		{ "LIGHTSALMON", LightSalmon },
		{ "LIGHTSEAGREEN", LightSeaGreen },
		{ "LIGHTSKYBLUE", LightSkyBlue },
		{ "LIGHTSLATEGRAY", LightSlateGray },
		{ "LIGHTSTEELBLUE", LightSteelBlue },
		{ "LIGHTYELLOW", LightYellow },
		{ "LIME", Lime },
		{ "LIMEGREEN", LimeGreen },
		{ "LINEN", Linen },
		{ "MAGENTA", Magenta },
		{ "MAROON", Maroon },
		{ "MEDIUMAQUAMARINE", MediumAquamarine },
		{ "MEDIUMBLUE", MediumBlue },
		{ "MEDIUMORCHID", MediumOrchid },
		{ "MEDIUMPURPLE", MediumPurple },
		{ "MEDIUMSEAGREEN", MediumSeaGreen },
		{ "MEDIUMSLATEBLUE", MediumSlateBlue },
		{ "MEDIUMSPRINGGREEN", MediumSpringGreen },
		{ "MEDIUMTURQUOISE", MediumTurquoise },
		{ "MEDIUMVIOLETRED", MediumVioletRed },
		{ "MIDNIGHTBLUE", MidnightBlue },
		{ "MINTCREAM", MintCream },
		{ "MISTYROSE", MistyRose },
		{ "MOCCASIN", Moccasin },
		{ "NAVAJOWHITE", NavajoWhite },
		{ "NAVYBLUE", NavyBlue },
		{ "OLDLACE", OldLace },
		{ "OLIVE", Olive },
		{ "OLIVEDRAB", OliveDrab },
		{ "ORANGE", Orange },
		{ "ORANGERED", OrangeRed },
		{ "ORCHID", Orchid },
		{ "PALEGOLDENROD", PaleGoldenrod },
		{ "PALEGREEN", PaleGreen },
		{ "PALETURQUOISE", PaleTurquoise },
		{ "PALEVIOLETRED", PaleVioletRed },
		{ "PAPAYAWHIP", PapayaWhip },
		{ "PEACHPUFF", PeachPuff },
		{ "PERU", Peru },
		{ "PINK", Pink },
		{ "PLUM", Plum },
		{ "POWDERBLUE", PowderBlue },
		{ "PURPLE", Purple },
		{ "REBECCAPURPLE", RebeccaPurple },
		{ "RED", Red },
		{ "ROSYBROWN", RosyBrown },
		{ "ROYALBLUE", RoyalBlue },
		{ "SADDLEBROWN", SaddleBrown },
		{ "SALMON", Salmon },
		{ "SANDYBROWN", SandyBrown },
		{ "SEAGREEN", SeaGreen },
		{ "SEASHELL", Seashell },
		{ "SIENNA", Sienna },
		{ "SILVER", Silver },
		{ "SKYBLUE", SkyBlue },
		{ "SLATEBLUE", SlateBlue },
		{ "SLATEGRAY", SlateGray },
		{ "SNOW", Snow },
		{ "SPRINGGREEN", SpringGreen },
		{ "STEELBLUE", SteelBlue },
		{ "TAN", Tan },
		{ "TEAL", Teal },
		{ "THISTLE", Thistle },
		{ "TOMATO", Tomato },
		{ "TRANSPARENT", Transparent },
		{ "TURQUOISE", Turquoise },
		{ "VIOLET", Violet },
		{ "WEBGRAY", WebGray },
		{ "WEBGREEN", WebGreen },
		{ "WEBMAROON", WebMaroon },
		{ "WEBPURPLE", WebPurple },
		{ "WHEAT", Wheat },
		{ "WHITE", White },
		{ "WHITESMOKE", WhiteSmoke },
		{ "YELLOW", Yellow },
		{ "YELLOWGREEN", YellowGreen }
	}.ToFrozenDictionary();

	public static Color AliceBlue => new Color(4042850303u);

	public static Color AntiqueWhite => new Color(4209760255u);

	public static Color Aqua => new Color(16777215u);

	public static Color Aquamarine => new Color(2147472639u);

	public static Color Azure => new Color(4043309055u);

	public static Color Beige => new Color(4126530815u);

	public static Color Bisque => new Color(4293182719u);

	public static Color Black => new Color(255u);

	public static Color BlanchedAlmond => new Color(4293643775u);

	public static Color Blue => new Color(65535u);

	public static Color BlueViolet => new Color(2318131967u);

	public static Color Brown => new Color(2771004159u);

	public static Color Burlywood => new Color(3736635391u);

	public static Color CadetBlue => new Color(1604231423u);

	public static Color Chartreuse => new Color(2147418367u);

	public static Color Chocolate => new Color(3530104575u);

	public static Color Coral => new Color(4286533887u);

	public static Color CornflowerBlue => new Color(1687547391u);

	public static Color Cornsilk => new Color(4294499583u);

	public static Color Crimson => new Color(3692313855u);

	public static Color Cyan => new Color(16777215u);

	public static Color DarkBlue => new Color(35839u);

	public static Color DarkCyan => new Color(9145343u);

	public static Color DarkGoldenrod => new Color(3095792639u);

	public static Color DarkGray => new Color(2846468607u);

	public static Color DarkGreen => new Color(6553855u);

	public static Color DarkKhaki => new Color(3182914559u);

	public static Color DarkMagenta => new Color(2332068863u);

	public static Color DarkOliveGreen => new Color(1433087999u);

	public static Color DarkOrange => new Color(4287365375u);

	public static Color DarkOrchid => new Color(2570243327u);

	public static Color DarkRed => new Color(2332033279u);

	public static Color DarkSalmon => new Color(3918953215u);

	public static Color DarkSeaGreen => new Color(2411499519u);

	public static Color DarkSlateBlue => new Color(1211993087u);

	public static Color DarkSlateGray => new Color(793726975u);

	public static Color DarkTurquoise => new Color(13554175u);

	public static Color DarkViolet => new Color(2483082239u);

	public static Color DeepPink => new Color(4279538687u);

	public static Color DeepSkyBlue => new Color(12582911u);

	public static Color DimGray => new Color(1768516095u);

	public static Color DodgerBlue => new Color(512819199u);

	public static Color Firebrick => new Color(2988581631u);

	public static Color FloralWhite => new Color(4294635775u);

	public static Color ForestGreen => new Color(579543807u);

	public static Color Fuchsia => new Color(4278255615u);

	public static Color Gainsboro => new Color(3705462015u);

	public static Color GhostWhite => new Color(4177068031u);

	public static Color Gold => new Color(4292280575u);

	public static Color Goldenrod => new Color(3668254975u);

	public static Color Gray => new Color(3200171775u);

	public static Color Green => new Color(16711935u);

	public static Color GreenYellow => new Color(2919182335u);

	public static Color Honeydew => new Color(4043305215u);

	public static Color HotPink => new Color(4285117695u);

	public static Color IndianRed => new Color(3445382399u);

	public static Color Indigo => new Color(1258324735u);

	public static Color Ivory => new Color(4294963455u);

	public static Color Khaki => new Color(4041641215u);

	public static Color Lavender => new Color(3873897215u);

	public static Color LavenderBlush => new Color(4293981695u);

	public static Color LawnGreen => new Color(2096890111u);

	public static Color LemonChiffon => new Color(4294626815u);

	public static Color LightBlue => new Color(2916673279u);

	public static Color LightCoral => new Color(4034953471u);

	public static Color LightCyan => new Color(3774873599u);

	public static Color LightGoldenrod => new Color(4210742015u);

	public static Color LightGray => new Color(3553874943u);

	public static Color LightGreen => new Color(2431553791u);

	public static Color LightPink => new Color(4290167295u);

	public static Color LightSalmon => new Color(4288707327u);

	public static Color LightSeaGreen => new Color(548580095u);

	public static Color LightSkyBlue => new Color(2278488831u);

	public static Color LightSlateGray => new Color(2005441023u);

	public static Color LightSteelBlue => new Color(2965692159u);

	public static Color LightYellow => new Color(4294959359u);

	public static Color Lime => new Color(16711935u);

	public static Color LimeGreen => new Color(852308735u);

	public static Color Linen => new Color(4210091775u);

	public static Color Magenta => new Color(4278255615u);

	public static Color Maroon => new Color(2955960575u);

	public static Color MediumAquamarine => new Color(1724754687u);

	public static Color MediumBlue => new Color(52735u);

	public static Color MediumOrchid => new Color(3126187007u);

	public static Color MediumPurple => new Color(2473647103u);

	public static Color MediumSeaGreen => new Color(1018393087u);

	public static Color MediumSlateBlue => new Color(2070474495u);

	public static Color MediumSpringGreen => new Color(16423679u);

	public static Color MediumTurquoise => new Color(1221709055u);

	public static Color MediumVioletRed => new Color(3340076543u);

	public static Color MidnightBlue => new Color(421097727u);

	public static Color MintCream => new Color(4127193855u);

	public static Color MistyRose => new Color(4293190143u);

	public static Color Moccasin => new Color(4293178879u);

	public static Color NavajoWhite => new Color(4292783615u);

	public static Color NavyBlue => new Color(33023u);

	public static Color OldLace => new Color(4260751103u);

	public static Color Olive => new Color(2155872511u);

	public static Color OliveDrab => new Color(1804477439u);

	public static Color Orange => new Color(4289003775u);

	public static Color OrangeRed => new Color(4282712319u);

	public static Color Orchid => new Color(3664828159u);

	public static Color PaleGoldenrod => new Color(4008225535u);

	public static Color PaleGreen => new Color(2566625535u);

	public static Color PaleTurquoise => new Color(2951671551u);

	public static Color PaleVioletRed => new Color(3681588223u);

	public static Color PapayaWhip => new Color(4293907967u);

	public static Color PeachPuff => new Color(4292524543u);

	public static Color Peru => new Color(3448061951u);

	public static Color Pink => new Color(4290825215u);

	public static Color Plum => new Color(3718307327u);

	public static Color PowderBlue => new Color(2967529215u);

	public static Color Purple => new Color(2686513407u);

	public static Color RebeccaPurple => new Color(1714657791u);

	public static Color Red => new Color(4278190335u);

	public static Color RosyBrown => new Color(3163525119u);

	public static Color RoyalBlue => new Color(1097458175u);

	public static Color SaddleBrown => new Color(2336560127u);

	public static Color Salmon => new Color(4202722047u);

	public static Color SandyBrown => new Color(4104413439u);

	public static Color SeaGreen => new Color(780883967u);

	public static Color Seashell => new Color(4294307583u);

	public static Color Sienna => new Color(2689740287u);

	public static Color Silver => new Color(3233857791u);

	public static Color SkyBlue => new Color(2278484991u);

	public static Color SlateBlue => new Color(1784335871u);

	public static Color SlateGray => new Color(1887473919u);

	public static Color Snow => new Color(4294638335u);

	public static Color SpringGreen => new Color(16744447u);

	public static Color SteelBlue => new Color(1182971135u);

	public static Color Tan => new Color(3535047935u);

	public static Color Teal => new Color(8421631u);

	public static Color Thistle => new Color(3636451583u);

	public static Color Tomato => new Color(4284696575u);

	public static Color Transparent => new Color(4294967040u);

	public static Color Turquoise => new Color(1088475391u);

	public static Color Violet => new Color(4001558271u);

	public static Color WebGray => new Color(2155905279u);

	public static Color WebGreen => new Color(8388863u);

	public static Color WebMaroon => new Color(2147483903u);

	public static Color WebPurple => new Color(2147516671u);

	public static Color Wheat => new Color(4125012991u);

	public static Color White => new Color(uint.MaxValue);

	public static Color WhiteSmoke => new Color(4126537215u);

	public static Color Yellow => new Color(4294902015u);

	public static Color YellowGreen => new Color(2597139199u);
}
