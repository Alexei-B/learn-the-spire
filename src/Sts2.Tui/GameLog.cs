using System;
using System.Collections.Generic;
using System.Linq;
using Sts2.Harness;
using Sts2.Localization;
using Color = Terminal.Gui.Color;

namespace Sts2.Tui;

/// <summary>
/// A running, scrolling log of what changed on each decision. Rather than plumb an event stream out
/// of the game, it derives events by diffing the immutable <see cref="GameState"/> before and after
/// each applied option: HP/gold/relics/potions/powers deltas, cards gained or moved between combat
/// piles (exhausted, generated into the draw pile, added to the deck), enemy defeats, and phase
/// transitions. The newest lines are at the bottom; <see cref="Render"/> returns the tail that fits.
/// </summary>
internal sealed class GameLog
{
    private readonly List<Line> _lines = new();

    public void Clear() => _lines.Clear();

    public void Note(string text) => _lines.Add(new Line().Dim(text));

    /// <summary>
    /// Record the consequences of one applied option: a header naming the action, then one line per
    /// observed change between <paramref name="before"/> and <paramref name="after"/>.
    /// </summary>
    public void Record(string header, GameState? before, GameState after)
    {
        var entries = new List<Line>();
        if (before is not null)
        {
            Diff(before, after, entries);
        }

        if (!string.IsNullOrWhiteSpace(header))
        {
            _lines.Add(new Line().Add("▸ ", Theme.Gold).T(Trim(header)));
        }
        foreach (Line e in entries)
        {
            _lines.Add(e);
        }

        // Cap the history so it can't grow without bound over a long run.
        const int cap = 500;
        if (_lines.Count > cap)
        {
            _lines.RemoveRange(0, _lines.Count - cap);
        }
    }

    private static string Trim(string s) => s.Length > 60 ? s.Substring(0, 59) + "…" : s;

    /// <summary>Word-wrap the log to the pane width and return the last <paramref name="height"/> rows.</summary>
    public IReadOnlyList<Line> Render(int width, int height)
    {
        if (width <= 0)
        {
            width = 30;
        }
        var wrapped = new List<Line>();
        foreach (Line l in _lines)
        {
            foreach (List<Seg> w in Markup.Wrap(l, Math.Max(8, width)))
            {
                var line = new Line();
                line.AddRange(w);
                wrapped.Add(line);
            }
        }
        if (height <= 0 || wrapped.Count <= height)
        {
            return wrapped;
        }
        return wrapped.GetRange(wrapped.Count - height, height);
    }

    // ---- Diffing ---------------------------------------------------------------

    private static void Diff(GameState before, GameState after, List<Line> log)
    {
        // Players are matched by NetId (order is stable, but be safe).
        foreach (PlayerState a in after.Players)
        {
            PlayerState? b = before.Players.FirstOrDefault(p => p.NetId == a.NetId);
            if (b is null)
            {
                continue;
            }
            DiffPlayer(b, a, log);
        }

        DiffEnemies(before, after, log);
        DiffPhase(before, after, log);
    }

    private static void DiffPlayer(PlayerState b, PlayerState a, List<Line> log)
    {
        string who = a.Character;

        if (a.CurrentHp < b.CurrentHp)
        {
            log.Add(Entry(Theme.Red, $"{who} took {b.CurrentHp - a.CurrentHp} damage → {a.CurrentHp} HP"));
        }
        else if (a.CurrentHp > b.CurrentHp)
        {
            log.Add(Entry(Theme.Green, $"{who} healed {a.CurrentHp - b.CurrentHp} → {a.CurrentHp} HP"));
        }
        if (a.MaxHp != b.MaxHp)
        {
            log.Add(Entry(Theme.Green, $"{who} max HP {(a.MaxHp > b.MaxHp ? "+" : "")}{a.MaxHp - b.MaxHp}"));
        }
        if (a.Gold != b.Gold)
        {
            log.Add(Entry(Theme.Gold, $"{(a.Gold > b.Gold ? "+" : "")}{a.Gold - b.Gold} gold → {a.Gold}g"));
        }

        foreach (string relic in Added(b.Relics, a.Relics))
        {
            log.Add(Entry(Theme.Teal, $"obtained {Localizer.RelicName(relic)}"));
        }
        foreach (string relic in Added(a.Relics, b.Relics))
        {
            log.Add(Entry(Theme.Dim, $"lost {Localizer.RelicName(relic)}"));
        }

        List<string> potBefore = b.Potions.Where(p => p is not null).Select(p => p!).ToList();
        List<string> potAfter = a.Potions.Where(p => p is not null).Select(p => p!).ToList();
        foreach (string pot in Added(potBefore, potAfter))
        {
            log.Add(Entry(Theme.Magenta, $"gained {Localizer.PotionName(pot)} potion"));
        }
        foreach (string pot in Added(potAfter, potBefore))
        {
            log.Add(Entry(Theme.Dim, $"used {Localizer.PotionName(pot)} potion"));
        }

        // The run deck (persists across combat) — reward/shop/event card gains and removals.
        foreach ((string name, int n) in CardDelta(b.Deck, a.Deck, positive: true))
        {
            log.Add(Entry(Theme.Green, $"added {name}{Times(n)} to deck"));
        }
        foreach ((string name, int n) in CardDelta(b.Deck, a.Deck, positive: false))
        {
            log.Add(Entry(Theme.Dim, $"removed {name}{Times(n)} from deck"));
        }

        DiffCombat(b.CombatState, a.CombatState, who, log);
    }

    private static void DiffCombat(PlayerCombatView? b, PlayerCombatView? a, string who, List<Line> log)
    {
        if (a is null)
        {
            return;
        }
        if (b is null)
        {
            return; // just entered combat; nothing to diff against yet
        }

        // Exhausted cards: additions to the exhaust pile.
        foreach ((string name, int n) in CardDelta(b.ExhaustPile, a.ExhaustPile, positive: true))
        {
            log.Add(Entry(Theme.Magenta, $"exhausted {name}{Times(n)}"));
        }

        // Cards newly created in combat (global count across all piles increased) — e.g. a Slimed
        // status shuffled into the draw pile. Attribute to the pile where they landed.
        Dictionary<string, int> allBefore = Counts(b.Hand.Concat(b.DrawPile).Concat(b.DiscardPile).Concat(b.ExhaustPile));
        Dictionary<string, int> allAfter = Counts(a.Hand.Concat(a.DrawPile).Concat(a.DiscardPile).Concat(a.ExhaustPile));
        foreach ((string name, int gained) in Delta(allBefore, allAfter))
        {
            string pile =
                Increased(b.DrawPile, a.DrawPile, name) ? "draw pile" :
                Increased(b.DiscardPile, a.DiscardPile, name) ? "discard pile" :
                Increased(b.Hand, a.Hand, name) ? "hand" : "play";
            log.Add(Entry(Theme.Green, $"gained {name}{Times(gained)} in {pile}"));
        }

        DiffPowers(b.Powers, a.Powers, who, log);
    }

    private static void DiffPowers(
        IReadOnlyList<PowerView> b, IReadOnlyList<PowerView> a, string who, List<Line> log)
    {
        Dictionary<string, int> bp = b.ToDictionary(p => p.PowerId, p => p.Amount);
        Dictionary<string, int> ap = a.ToDictionary(p => p.PowerId, p => p.Amount);
        foreach (string id in ap.Keys.Union(bp.Keys))
        {
            int before = bp.GetValueOrDefault(id);
            int after = ap.GetValueOrDefault(id);
            if (after == before)
            {
                continue;
            }
            string name = Localizer.PowerName(id);
            if (after == 0)
            {
                log.Add(Entry(Theme.Dim, $"{who} lost {name}"));
            }
            else if (before == 0)
            {
                log.Add(Entry(Theme.Blue, $"{who} gained {name} {after}"));
            }
            else
            {
                log.Add(Entry(Theme.Blue, $"{who} {name} {before}→{after}"));
            }
        }
    }

    private static void DiffEnemies(GameState before, GameState after, List<Line> log)
    {
        if (before.Combat is not { } bc)
        {
            return;
        }
        IReadOnlyList<EnemyView> afterEnemies = after.Combat?.Enemies ?? new List<EnemyView>();
        foreach (EnemyView e in bc.Enemies)
        {
            EnemyView? now = afterEnemies.FirstOrDefault(x => x.CombatId == e.CombatId);
            bool wasAlive = e.CurrentHp > 0;
            bool dead = now is null || now.CurrentHp <= 0;
            if (wasAlive && dead)
            {
                log.Add(Entry(Theme.Teal, $"defeated {Localizer.MonsterName(e.MonsterId)}"));
            }
        }
    }

    private static void DiffPhase(GameState before, GameState after, List<Line> log)
    {
        if (before.Phase == after.Phase)
        {
            return;
        }
        // Only announce meaningful room/screen transitions (combat entry/exit, arriving at a screen).
        string? note = after.Phase switch
        {
            GamePhase.Combat when before.Phase != GamePhase.Choice => "⚔ combat begins",
            GamePhase.Reward when before.Phase is GamePhase.Combat or GamePhase.Choice => "✔ combat won",
            GamePhase.Event => "entered an event",
            GamePhase.Shop => "entered the shop",
            GamePhase.RestSite => "reached a rest site",
            GamePhase.Treasure => "opened a treasure chest",
            GamePhase.GameOver => after.IsVictory ? "★ VICTORY" : "☠ defeated",
            _ => null,
        };
        if (note is not null)
        {
            Color c = after.Phase == GamePhase.GameOver
                ? (after.IsVictory ? Theme.Green : Theme.Red)
                : Theme.Teal;
            log.Add(Entry(c, note));
        }
        if (after.ActIndex > before.ActIndex)
        {
            log.Add(Entry(Theme.Gold, $"advanced to Act {after.ActIndex + 1}"));
        }
    }

    // ---- helpers ---------------------------------------------------------------

    private static Line Entry(Color color, string text) => new Line().Add("  · ", Theme.Dim).Add(text, color);

    private static string Times(int n) => n > 1 ? $" x{n}" : "";

    private static string CardName(CardView c) => Localizer.CardName(c.CardId) + (c.Upgraded ? "+" : "");

    private static Dictionary<string, int> Counts(IEnumerable<CardView> cards)
    {
        var d = new Dictionary<string, int>();
        foreach (CardView c in cards)
        {
            string n = CardName(c);
            d[n] = d.GetValueOrDefault(n) + 1;
        }
        return d;
    }

    private static bool Increased(IReadOnlyList<CardView> before, IReadOnlyList<CardView> after, string name)
    {
        int b = before.Count(c => CardName(c) == name);
        int a = after.Count(c => CardName(c) == name);
        return a > b;
    }

    /// <summary>Per-card count gained (positive:true) or lost (positive:false) between two piles.</summary>
    private static IEnumerable<(string name, int count)> CardDelta(
        IReadOnlyList<CardView> before, IReadOnlyList<CardView> after, bool positive)
    {
        Dictionary<string, int> bc = Counts(before);
        Dictionary<string, int> ac = Counts(after);
        return positive ? Delta(bc, ac) : Delta(ac, bc);
    }

    /// <summary>Names whose count is higher in <paramref name="hi"/> than in <paramref name="lo"/>, with the gain.</summary>
    private static IEnumerable<(string name, int count)> Delta(Dictionary<string, int> lo, Dictionary<string, int> hi)
    {
        foreach (string name in hi.Keys)
        {
            int diff = hi[name] - lo.GetValueOrDefault(name);
            if (diff > 0)
            {
                yield return (name, diff);
            }
        }
    }

    /// <summary>Items present in <paramref name="after"/> more times than in <paramref name="before"/> (multiset add).</summary>
    private static IEnumerable<string> Added(IReadOnlyList<string> before, IReadOnlyList<string> after)
    {
        var counts = new Dictionary<string, int>();
        foreach (string s in before)
        {
            counts[s] = counts.GetValueOrDefault(s) + 1;
        }
        foreach (string s in after)
        {
            if (counts.GetValueOrDefault(s) > 0)
            {
                counts[s]--;
            }
            else
            {
                yield return s;
            }
        }
    }
}
