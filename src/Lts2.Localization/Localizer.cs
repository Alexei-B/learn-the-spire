using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;

namespace Lts2.Localization;

/// <summary>
/// Resolves human-readable names and descriptions for game content (cards, relics, potions, powers,
/// events) by reusing the game's own localization pipeline — the real <c>LocString</c> +
/// SmartFormat rendering, so descriptions carry the actual numbers.
///
/// This is opt-in and decoupled from the harness: it works only when the English localization
/// tables have been placed at the shim's <c>res://localization</c> root (this library packages them
/// there — see the .csproj; run <c>scripts/extract-localization.ps1</c> to produce them). When the
/// tables are absent, <see cref="Available"/> is false and every method returns the model id, so the
/// harness never depends on the (gitignored) game content.
///
/// Callers pass model ids (the read-model uses ids, not live models); lookups resolve the canonical
/// model via <c>ModelDb</c>. The game must already be booted (<c>GameRuntime.EnsureInitialized</c>).
/// </summary>
public static class Localizer
{
    private static bool _confirmed;

    /// <summary>
    /// True once the real localization tables are loaded (probed via a known key). While false it is
    /// re-checked on each access, so it flips to true as soon as the game is booted with the tables
    /// present; once true it stays true.
    /// </summary>
    public static bool Available
    {
        get
        {
            if (_confirmed)
            {
                return true;
            }
            try
            {
                _confirmed = LocString.Exists("cards", "BASH.title");
            }
            catch
            {
                _confirmed = false;
            }
            return _confirmed;
        }
    }

    // Every method localizes, then falls back to the model id (names) or empty (descriptions) when
    // the lookup fails, is missing, or renders to whitespace — there is no global gate, so it works
    // the moment the tables are present and degrades cleanly when they are not.

    // ---- Cards -----------------------------------------------------------------

    public static string CardName(string id) =>
        Cards.TryGetValue(id, out CardModel? c) ? Name(c.TitleLocString, id) : id;

    public static string CardDescription(string id)
    {
        if (!Cards.TryGetValue(id, out CardModel? c))
        {
            return string.Empty;
        }
        try
        {
            if (!c.Description.Exists())
            {
                return string.Empty;
            }
            // Raw (markup kept: colours + energy icons) — the caller renders it.
            string s = c.GetDescriptionForPile(PileType.None);
            return string.IsNullOrWhiteSpace(s) ? string.Empty : s;
        }
        catch
        {
            return string.Empty;
        }
    }

    // ---- Relics ----------------------------------------------------------------

    public static string RelicName(string id) =>
        Relics.TryGetValue(id, out RelicModel? r) ? Name(r.Title, id) : id;

    public static string RelicDescription(string id) =>
        Relics.TryGetValue(id, out RelicModel? r) ? Desc(r.DynamicDescription) : string.Empty;

    // ---- Potions ---------------------------------------------------------------

    public static string PotionName(string id) =>
        Potions.TryGetValue(id, out PotionModel? p) ? Name(p.Title, id) : id;

    public static string PotionDescription(string id) =>
        Potions.TryGetValue(id, out PotionModel? p) ? Desc(p.DynamicDescription) : string.Empty;

    // ---- Powers ----------------------------------------------------------------

    public static string PowerName(string id) =>
        Powers.TryGetValue(id, out PowerModel? p) ? Name(p.Title, id) : id;

    // ---- Monsters --------------------------------------------------------------

    public static string MonsterName(string id) =>
        Monsters.TryGetValue(id, out MonsterModel? m) ? Name(m.Title, id) : id;

    // ---- Encounters (boss / combat names) --------------------------------------

    public static string EncounterName(string id) =>
        Encounters.TryGetValue(id, out EncounterModel? e) ? Name(e.Title, id) : id;

    // ---- Events ----------------------------------------------------------------

    public static string EventName(string id) =>
        Events.TryGetValue(id, out EventModel? e) ? Name(e.Title, id) : id;

    /// <summary>
    /// Localize one event option's title from the event id and the option's text key, or null when it
    /// has no localized title (e.g. the Neow ancient's relic options use dynamic titles — the caller
    /// should fall back to the relic name).
    /// </summary>
    public static string? EventOptionTitle(string eventId, string textKey) =>
        Events.TryGetValue(eventId, out EventModel? ev) ? NameOrNull(() => ev.GetOptionTitle(textKey)) : null;

    /// <summary>Localize one event option's description (raw markup kept), or null when there is none.</summary>
    public static string? EventOptionDescription(string eventId, string textKey) =>
        Events.TryGetValue(eventId, out EventModel? ev) ? DescOrNull(() => ev.GetOptionDescription(textKey)) : null;

    // ---- BBCode stripping ------------------------------------------------------

    private static readonly Regex ImgTag = new(@"\[img\].*?\[/img\]", RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex AnyTag = new(@"\[/?[a-zA-Z][^\]]*\]", RegexOptions.Compiled);

    /// <summary>Strip the game's BBCode-style markup (<c>[gold]…[/gold]</c>, <c>[img]…[/img]</c>) for plain display.</summary>
    public static string Clean(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }
        text = ImgTag.Replace(text, string.Empty);
        text = AnyTag.Replace(text, string.Empty);
        return text.Trim();
    }

    // ---- Render helpers (localize, else fall back) + lazy id→model indexes -----

    /// <summary>Render a name LocString, falling back to <paramref name="id"/> on miss/whitespace/throw.</summary>
    private static string Name(LocString? ls, string id)
    {
        try
        {
            if (ls is null || !ls.Exists())
            {
                return id;
            }
            string s = Clean(ls.GetFormattedText());
            return string.IsNullOrWhiteSpace(s) ? id : s;
        }
        catch
        {
            return id;
        }
    }

    /// <summary>Render a description LocString (raw markup kept), falling back to empty on miss/whitespace/throw.</summary>
    private static string Desc(LocString? ls)
    {
        try
        {
            if (ls is null || !ls.Exists())
            {
                return string.Empty;
            }
            string s = ls.GetFormattedText();
            return string.IsNullOrWhiteSpace(s) ? string.Empty : s;
        }
        catch
        {
            return string.Empty;
        }
    }

    /// <summary>Render an optional description LocString to raw text, or null when missing/whitespace/throwing.</summary>
    private static string? DescOrNull(Func<LocString?> get)
    {
        try
        {
            LocString? ls = get();
            if (ls is null || !ls.Exists())
            {
                return null;
            }
            string s = ls.GetFormattedText();
            return string.IsNullOrWhiteSpace(s) ? null : s;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Render an optional LocString to text, or null when missing/whitespace/throwing.</summary>
    private static string? NameOrNull(Func<LocString?> get)
    {
        try
        {
            LocString? ls = get();
            if (ls is null || !ls.Exists())
            {
                return null;
            }
            string s = Clean(ls.GetFormattedText());
            return string.IsNullOrWhiteSpace(s) ? null : s;
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<string, CardModel>? _cards;
    private static Dictionary<string, RelicModel>? _relics;
    private static Dictionary<string, PotionModel>? _potions;
    private static Dictionary<string, PowerModel>? _powers;
    private static Dictionary<string, EventModel>? _events;

    private static Dictionary<string, MonsterModel>? _monsters;
    private static Dictionary<string, EncounterModel>? _encounters;

    private static Dictionary<string, CardModel> Cards => _cards ??= Index(ModelDb.AllCards);
    private static Dictionary<string, RelicModel> Relics => _relics ??= Index(ModelDb.AllRelics);
    private static Dictionary<string, PotionModel> Potions => _potions ??= Index(ModelDb.AllPotions);
    private static Dictionary<string, PowerModel> Powers => _powers ??= Index(ModelDb.AllPowers);
    private static Dictionary<string, MonsterModel> Monsters => _monsters ??= Index(ModelDb.Monsters);
    private static Dictionary<string, EncounterModel> Encounters => _encounters ??= Index(ModelDb.AllEncounters);

    private static Dictionary<string, EventModel> Events =>
        _events ??= Index(ModelDb.AllEvents.Concat(ModelDb.AllAncients.Cast<EventModel>()));

    private static Dictionary<string, T> Index<T>(IEnumerable<T> models) where T : AbstractModel
    {
        var d = new Dictionary<string, T>();
        foreach (T m in models)
        {
            d[m.Id.Entry] = m; // later wins; canonical models are unique by entry
        }
        return d;
    }
}
