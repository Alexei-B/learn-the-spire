using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Rewards;
using Lts2.Harness;
using Lts2.Localization;
using Terminal.Gui;

namespace Lts2.Tui;

/// <summary>Builds the coloured board lines and the plain option labels from the read-model.</summary>
internal static class BoardRenderer
{
    // ---- Board (left/centre canvas) --------------------------------------------

    public static List<Line> Board(GameState state, int width, int height = 1000, Coord? mapHighlight = null)
    {
        if (width <= 0)
        {
            width = 100;
        }
        var lines = new List<Line> { StatusLine(state), new Line() };

        switch (state.Phase)
        {
            case GamePhase.Combat:
            case GamePhase.Choice:
                CombatBoard(state, width, lines);
                if (state.Phase == GamePhase.Choice && state.PendingChoice is { } pc)
                {
                    ChoiceLines(pc, lines);
                }
                break;
            case GamePhase.Reward:
                Rewards(state, lines);
                break;
            case GamePhase.Event:
                Event(state, lines, width);
                break;
            case GamePhase.Treasure:
                Treasure(state, lines);
                break;
            case GamePhase.RestSite:
                RestSite(state, lines);
                break;
            case GamePhase.Shop:
                Shop(state, lines);
                break;
            case GamePhase.CrystalSphere:
                CrystalSphere(state, lines);
                break;
            case GamePhase.GameOver:
                GameOver(state, lines);
                break;
            default:
                if (state.Map is { } map)
                {
                    MapLines(map, lines, width, height, mapHighlight);
                }
                break;
        }
        return lines;
    }

    /// <summary>The right-hand side panel: the piles in combat, the act map (with connections) elsewhere.</summary>
    public static List<Line> SidePanel(GameState state, int width, int height = 1000, Coord? mapHighlight = null)
    {
        if (width <= 0)
        {
            width = 30;
        }
        var lines = new List<Line>();
        if (state.Phase is GamePhase.Combat or GamePhase.Choice)
        {
            Piles(state, lines);
        }
        else if (state.Map is { } map)
        {
            MapLines(map, lines, width, height, mapHighlight);
        }
        return lines;
    }

    private static void Piles(GameState state, List<Line> lines)
    {
        PlayerCombatView? cs = state.Players.Select(p => p.CombatState).FirstOrDefault(c => c is not null);
        if (cs is null)
        {
            return;
        }
        PileList(lines, "DRAW", cs.DrawPile, Theme.Teal);
        lines.Add(new Line());
        PileList(lines, "DISCARD", cs.DiscardPile, Theme.Gold);
        lines.Add(new Line());
        PileList(lines, "EXHAUST", cs.ExhaustPile, Theme.Magenta);
    }

    private static void PileList(List<Line> lines, string title, IReadOnlyList<CardView> pile, Color color)
    {
        lines.Add(new Line().Add($"{title} ({pile.Count})", color));
        foreach (var grp in pile.GroupBy(c => (CardName(c), ModifierKey(c))).OrderBy(g => g.Key.Item1))
        {
            var l = new Line().Dim("  ").T(grp.Key.Item1);
            AppendModifierSegs(l, grp.First());
            if (grp.Count() > 1)
            {
                l.Dim($" x{grp.Count()}");
            }
            lines.Add(l);
        }
        if (pile.Count == 0)
        {
            lines.Add(new Line().Dim("  (empty)"));
        }
    }

    private static Line StatusLine(GameState state)
    {
        PlayerState p = state.Players[0];
        var l = new Line();
        l.Add($"Act {state.ActIndex + 1}", Theme.Gold).Dim(" · ");
        l.T($"Floor {state.Floor}").Dim(" · ");
        l.Add($"{p.CurrentHp}/{p.MaxHp} HP", Theme.Hp(p.CurrentHp, p.MaxHp)).Dim(" · ");
        l.Add($"{p.Gold}g", Theme.Gold).Dim(" · ");
        l.Add(p.Character, Theme.Green).Dim(" · ");
        l.T($"A{state.AscensionLevel}").Dim(" · ");
        l.T($"Score {state.Score}").Dim(" · ");
        l.Add(state.Phase.ToString(), Theme.Teal);
        return l;
    }

    // ---- Combat ----------------------------------------------------------------

    // Combat: allies (players + Osty) drawn as boxes on the left, enemies as boxes on the right, each
    // with a coloured health bar, an info line (intent / energy+hand) and its powers. Then the hand.
    private static void CombatBoard(GameState state, int width, List<Line> lines)
    {
        if (state.Combat is not { } combat)
        {
            return;
        }
        int gap = 2;
        int colW = Math.Clamp((width - gap) / 2, 16, 44);

        var left = new List<Line>();
        foreach (PlayerState p in state.Players)
        {
            if (p.CombatState is not { } cs)
            {
                continue;
            }
            var info = new List<Seg> { new(" Energy ", Theme.Dim) };
            info.AddRange(Markup.EnergyCircles(cs.Energy, cs.MaxEnergy));
            // The Regent's star power sits between energy and hand size; hidden at 0.
            if (cs.Stars > 0)
            {
                info.Add(new Seg($"  ★{cs.Stars}", Theme.Gold));
            }
            info.Add(new Seg($"  Hand {cs.Hand.Count}", Theme.Dim));
            // The Defect's orb slots (if any) render as their own line below energy/hand, before powers.
            IReadOnlyList<Seg>? orbInfo = cs.OrbSlots > 0 ? OrbSegs(cs) : null;
            left.AddRange(CreatureBox(p.Character, p.CurrentHp, p.MaxHp, p.Block, cs.Powers, info, colW, orbInfo));
            left.Add(new Line());
            if (cs.Osty is { IsAlive: true } osty)
            {
                left.AddRange(CreatureBox("Osty", osty.CurrentHp, osty.MaxHp, osty.Block, osty.Powers, null, colW));
                left.Add(new Line());
            }
        }

        var right = new List<Line>();
        foreach (EnemyView e in combat.Enemies)
        {
            string name = $"#{e.CombatId} {Localizer.MonsterName(e.MonsterId)}";
            var info = new List<Seg> { new(" Intent: ", Theme.Dim) };
            info.AddRange(IntentSegs(e));
            right.AddRange(CreatureBox(name, e.CurrentHp, e.MaxHp, e.Block, e.Powers, info, colW));
            right.Add(new Line());
        }

        int rows = Math.Max(left.Count, right.Count);
        for (int r = 0; r < rows; r++)
        {
            var l = new Line();
            AppendPadded(l, r < left.Count ? left[r] : null, colW);
            l.T(new string(' ', gap));
            AppendPadded(l, r < right.Count ? right[r] : null, colW);
            lines.Add(l);
        }

        // The full hand below (the decisions panel only lists playable cards).
        PlayerCombatView? me = state.Players[0].CombatState;
        if (me is not null)
        {
            lines.Add(new Line());
            lines.Add(new Line().Add("HAND", Theme.Gold));
            if (me.Hand.Count == 0)
            {
                lines.Add(new Line().Dim("  (empty)"));
            }
            foreach (CardView c in me.Hand)
            {
                lines.Add(CardLine(c));
            }
        }
    }

    private static int PowerAmount(IReadOnlyList<PowerView> powers, string id) =>
        powers.FirstOrDefault(p => p.PowerId == id)?.Amount ?? 0;

    /// <summary>
    /// A bordered creature panel: name, coloured health bar, an info line, and its powers. The border
    /// is red normally, green if it would die to poison this turn, purple if to doom, grey if it has block.
    /// </summary>
    private static List<Line> CreatureBox(
        string name, int cur, int max, int block, IReadOnlyList<PowerView> powers, IReadOnlyList<Seg>? info, int w,
        IReadOnlyList<Seg>? orbInfo = null)
    {
        int poison = PowerAmount(powers, "POISON_POWER");
        int doom = PowerAmount(powers, "DOOM_POWER");
        int inner = w - 2;
        Color border =
            poison > 0 && poison >= cur ? Theme.Green :
            doom > 0 && doom >= cur ? Theme.Magenta :
            block > 0 ? Theme.LightGrey :
            Theme.Red;

        var lines = new List<Line>
        {
            new Line().Add("┌" + new string('─', inner) + "┐", border),
            BoxLine(new List<Seg> { new(" " + name, Theme.Fg) }, inner, border),
        };

        string hpText = block > 0 ? $" {cur}/{max} +{block}" : $" {cur}/{max}";
        int barCells = Math.Max(4, inner - hpText.Length - 1);
        var bar = HealthBar(cur, max, poison, doom, barCells);
        bar.Add(new Seg(hpText, block > 0 ? Theme.Blue : Theme.Fg));
        lines.Add(BoxLine(bar, inner, border));

        if (info is { Count: > 0 })
        {
            lines.Add(BoxLine(new List<Seg>(info), inner, border));
        }
        if (orbInfo is { Count: > 0 })
        {
            lines.Add(BoxLine(new List<Seg>(orbInfo), inner, border));
        }
        if (powers.Count > 0)
        {
            lines.Add(BoxLine(new List<Seg> { new(" " + PowerText(powers), Theme.Dim) }, inner, border));
        }
        lines.Add(new Line().Add("└" + new string('─', inner) + "┘", border));
        return lines;
    }

    // The orb queue as a single info line: each filled slot shows the orb's name and its key number
    // (Lightning/Frost/Plasma/Glass show the per-turn passive value; a Dark orb shows its accumulated
    // charge), empty slots show a dot. Coloured per orb type.
    private static List<Seg> OrbSegs(PlayerCombatView cs)
    {
        var segs = new List<Seg> { new(" Orbs ", Theme.Dim) };
        for (int i = 0; i < cs.OrbSlots; i++)
        {
            segs.Add(new Seg(i == 0 ? "" : " ", Theme.Fg));
            if (i < cs.Orbs.Count)
            {
                OrbView o = cs.Orbs[i];
                int n = o.OrbId == "DARK_ORB" ? o.EvokeValue : o.PassiveValue;
                segs.Add(new Seg($"{OrbName(o.OrbId)} {n}", OrbColor(o.OrbId)));
            }
            else
            {
                segs.Add(new Seg("·", Theme.Dim));
            }
        }
        return segs;
    }

    private static string OrbName(string id)
    {
        string s = id.EndsWith("_ORB", StringComparison.Ordinal) ? id[..^4] : id;
        return System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase(s.Replace('_', ' ').ToLowerInvariant());
    }

    private static Color OrbColor(string id) => id switch
    {
        "LIGHTNING_ORB" => Theme.Gold,
        "FROST_ORB" => Theme.Blue,
        "DARK_ORB" => Theme.Magenta,
        "PLASMA_ORB" => Theme.Teal,
        "GLASS_ORB" => Theme.LightGrey,
        _ => Theme.Fg,
    };

    // A health bar: red current HP, purple the doom threshold, green the HP poison will remove this
    // turn, and dark for already-lost HP.
    private static List<Seg> HealthBar(int cur, int max, int poison, int doom, int cells)
    {
        if (max <= 0)
        {
            max = 1;
        }
        cur = Math.Clamp(cur, 0, max);
        int filled = cur <= 0 ? 0 : Math.Max(1, (int)Math.Round((double)cur / max * cells));
        int doomCells = doom <= 0 ? 0 : (int)Math.Round((double)Math.Min(doom, max) / max * cells);
        int poisonCells = poison <= 0 ? 0 : (int)Math.Round((double)Math.Min(poison, cur) / max * cells);

        var segs = new List<Seg>();
        var sb = new System.Text.StringBuilder();
        Color? run = null;
        void Flush()
        {
            if (sb.Length > 0 && run.HasValue)
            {
                segs.Add(new Seg(sb.ToString(), run.Value));
                sb.Clear();
            }
        }
        for (int i = 0; i < cells; i++)
        {
            Color c =
                i >= filled ? Theme.HpLost :
                i < doomCells ? Theme.Magenta :
                i >= filled - poisonCells ? Theme.Green :
                Theme.Red;
            if (run != c)
            {
                Flush();
                run = c;
            }
            sb.Append('█');
        }
        Flush();
        return segs;
    }

    // A content row inside a box: left border, content padded/truncated to inner width, right border.
    private static Line BoxLine(List<Seg> content, int inner, Color border)
    {
        var l = new Line().Add("│", border);
        int used = 0;
        foreach (Seg s in content)
        {
            if (used >= inner)
            {
                break;
            }
            string t = s.Text.Length > inner - used ? s.Text.Substring(0, inner - used) : s.Text;
            l.Add(t, s.Fg);
            used += t.Length;
        }
        if (used < inner)
        {
            l.T(new string(' ', inner - used));
        }
        return l.Add("│", border);
    }

    // Append a (fixed-width box) source line to a target line, padding to exactly width columns.
    private static void AppendPadded(Line target, Line? src, int width)
    {
        int used = 0;
        if (src is not null)
        {
            foreach (Seg s in src)
            {
                if (used >= width)
                {
                    break;
                }
                string t = s.Text.Length > width - used ? s.Text.Substring(0, width - used) : s.Text;
                target.Add(t, s.Fg);
                used += t.Length;
            }
        }
        if (used < width)
        {
            target.T(new string(' ', width - used));
        }
    }

    private static string CardName(CardView c) => Localizer.CardName(c.CardId) + (c.Upgraded ? "+" : "");

    /// <summary>Green when buffed above the printed value, red when weakened below it, neutral when equal.</summary>
    private static Color DeltaColor(int actual, int baseline) =>
        actual > baseline ? Theme.Green : actual < baseline ? Theme.Red : Theme.Fg;

    /// <summary>
    /// The card's live attack/block preview (in combat): the actual number after all powers, coloured
    /// green if it's more than the printed value, red if less. Appended to card labels and the hand.
    /// </summary>
    private static void AppendEffectSegs(List<Seg> segs, CardView c)
    {
        // The Regent's star cost (a second resource paid alongside energy); shown only when positive.
        if (c.StarCost > 0)
        {
            segs.Add(new Seg($"  ★{c.StarCost}", Theme.Gold));
        }
        if (c.Damage is { } dmg && c.BaseDamage is { } bd)
        {
            segs.Add(new Seg("   ", Theme.Dim));
            segs.Add(new Seg(dmg.ToString(), DeltaColor(dmg, bd)));
            segs.Add(new Seg(" dmg", Theme.Dim));
        }
        if (c.Block is { } blk && c.BaseBlock is { } bb)
        {
            segs.Add(new Seg("  +", Theme.Dim));
            segs.Add(new Seg(blk.ToString(), DeltaColor(blk, bb)));
            segs.Add(new Seg(" blk", Theme.Dim));
        }
    }

    /// <summary>
    /// Modifiers applied to the card beyond its printed form — an enchantment (purple), an affliction
    /// like Bound (red), a granted Replay count, and granted keywords (Retain from Transfigure, …) —
    /// so they're visible on the card wherever it's listed. Nothing is appended for a plain card.
    /// </summary>
    private static void AppendModifierSegs(List<Seg> segs, CardView c)
    {
        if (c.EnchantmentId is { } e)
        {
            segs.Add(new Seg($"  [{Localizer.EnchantmentName(e)}]", Theme.Magenta));
        }
        if (c.AfflictionId is { } a)
        {
            segs.Add(new Seg($"  [{Localizer.AfflictionName(a)}]", Theme.Red));
        }
        if (c.ReplayCount > 0)
        {
            segs.Add(new Seg($"  Replay {c.ReplayCount}", Theme.Teal));
        }
        foreach (string kw in c.AddedKeywords)
        {
            segs.Add(new Seg($"  {kw}", Theme.Teal));
        }
    }

    // A stable signature of a card's modifiers, so pile/deck grouping keeps differently-modified copies
    // of the same card on separate lines.
    private static string ModifierKey(CardView c) =>
        $"{c.EnchantmentId}|{c.AfflictionId}|{c.ReplayCount}|{string.Join(",", c.AddedKeywords)}";

    private static Line CardLine(CardView c)
    {
        var l = new Line();
        l.Add(c.CanPlay ? "  ▸ " : "    ", c.CanPlay ? Theme.Green : Theme.Dim);
        Seg cost = Markup.Cost(c.CostsX, c.EnergyCost);
        l.Add(cost.Text, cost.Fg);
        l.T(" ");
        l.Add(CardName(c), c.CanPlay ? Theme.Fg : Theme.Dim);
        AppendEffectSegs(l, c);
        AppendModifierSegs(l, c);
        l.Dim($"  {c.Type}/{c.Rarity}");
        return l;
    }

    private static void ChoiceLines(PendingChoiceView pc, List<Line> lines)
    {
        lines.Add(new Line());
        string heading = pc.IsUpgradeSelection
            ? "CHOOSE A CARD TO FORGE (shown upgraded)"
            : $"CHOOSE {pc.MinSelect}-{pc.MaxSelect} CARD(S)";
        lines.Add(new Line().Add(heading, Theme.Teal));
        foreach (CardView c in pc.Options)
        {
            lines.Add(CardLine(c));
        }
    }

    // The enemy's telegraphed intent as coloured segments: attack damage is the actual incoming
    // damage after modifiers (enemy Strength/Weak, the player's Vulnerable/…), coloured green when it's
    // more than the base and red when less — same convention as the player's own attack previews.
    private static List<Seg> IntentSegs(EnemyView e)
    {
        var segs = new List<Seg>();
        if (e.Intents.Count == 0)
        {
            segs.Add(new Seg("?", Theme.Red));
            return segs;
        }
        for (int idx = 0; idx < e.Intents.Count; idx++)
        {
            IntentView i = e.Intents[idx];
            if (idx > 0)
            {
                segs.Add(new Seg(", ", Theme.Dim));
            }
            if (i.Damage is { } dmg)
            {
                Color dc = i.BaseDamage is { } bd ? DeltaColor(dmg, bd) : Theme.Red;
                segs.Add(new Seg($"{i.Type} ", Theme.Red));
                segs.Add(new Seg(dmg.ToString(), dc));
                if ((i.Hits ?? 1) > 1)
                {
                    segs.Add(new Seg($"x{i.Hits}", Theme.Red));
                }
            }
            else
            {
                segs.Add(new Seg(i.Type.ToString(), Theme.Red));
            }
        }
        return segs;
    }

    private static string PowerText(IReadOnlyList<PowerView> powers) =>
        string.Join("  ", powers.Select(pw => $"{Localizer.PowerName(pw.PowerId)} {pw.Amount}"));

    // ---- Rewards / Event / Treasure / Rest / Shop ------------------------------

    private static void Rewards(GameState state, List<Line> lines)
    {
        if (state.Rewards is not { } r)
        {
            return;
        }
        lines.Add(new Line().Add("REWARDS", Theme.Gold));
        foreach (RewardView rw in r.Rewards)
        {
            string detail = rw.Type switch
            {
                RewardType.Gold => $"{rw.Gold} gold",
                RewardType.Potion => Localizer.PotionName(rw.PotionId ?? "?"),
                RewardType.Relic => Localizer.RelicName(rw.RelicId ?? "?"),
                RewardType.Card => string.Join(", ", (rw.Cards ?? new List<CardView>()).Select(CardName)),
                _ => "-",
            };
            var l = new Line().Add($"  {rw.Type}: ", Theme.Teal).T(detail);
            if (rw.Type == RewardType.Card && rw.CardAlternatives is { Count: > 0 } alts)
            {
                l.Dim($"  (alt: {string.Join("/", alts)})");
            }
            if (rw.Taken)
            {
                l.Add("  [taken]", Theme.Dim);
            }
            lines.Add(l);
        }
    }

    private static void Event(GameState state, List<Line> lines, int width)
    {
        if (state.Event is not { } ev)
        {
            return;
        }
        var head = new Line().Add(Localizer.EventName(ev.EventId), Theme.Teal);
        if (ev.IsAncient)
        {
            head.Dim("  (ancient)");
        }
        lines.Add(head);

        // The event's body/flavour text (raw game markup, wrapped to the board width).
        if (!string.IsNullOrWhiteSpace(ev.Description))
        {
            lines.Add(new Line());
            foreach (List<Seg> wrapped in Markup.Wrap(Markup.Parse(ev.Description, Theme.Fg), Math.Max(20, width - 2)))
            {
                lines.Add(WrapLine(wrapped));
            }
        }

        lines.Add(new Line());
        foreach (EventOptionView o in ev.Options)
        {
            string title = EventTitle(ev.EventId, o, o.TextKey, o.RelicId);
            var l = new Line().Add("  • ", Theme.Gold).T(title);
            if (o.RelicId is not null && !title.Contains(Localizer.RelicName(o.RelicId)))
            {
                l.Add($"  (relic {Localizer.RelicName(o.RelicId)})", Theme.Teal);
            }
            lines.Add(l);

            // The option's outcome text under it (raw markup, wrapped & indented).
            string desc = EventDesc(ev.EventId, o, o.TextKey, o.RelicId);
            if (!string.IsNullOrWhiteSpace(desc))
            {
                foreach (List<Seg> wrapped in Markup.Wrap(Markup.Parse(desc, Theme.Dim), Math.Max(20, width - 6)))
                {
                    lines.Add(WrapLine(wrapped, "      "));
                }
            }
        }
    }

    /// <summary>Wrap a run of wrapped segments into a <see cref="Line"/> with an optional indent prefix.</summary>
    private static Line WrapLine(List<Seg> segs, string indent = "  ")
    {
        var l = new Line().Dim(indent);
        l.AddRange(segs);
        return l;
    }

    private static void Treasure(GameState state, List<Line> lines)
    {
        if (state.Treasure is not { } tr)
        {
            return;
        }
        lines.Add(new Line().Add("TREASURE", Theme.Gold));
        if (tr.Relics.Count == 0)
        {
            lines.Add(new Line().Dim("  (empty)"));
        }
        foreach (string relic in tr.Relics)
        {
            lines.Add(new Line().Add($"  {Localizer.RelicName(relic)}", Theme.Teal));
        }
    }

    private static void RestSite(GameState state, List<Line> lines)
    {
        if (state.RestSite is not { } rs)
        {
            return;
        }
        lines.Add(new Line().Add("REST SITE", Theme.Green));
        foreach (RestSiteOptionView o in rs.Options)
        {
            lines.Add(new Line().Add($"  {o.OptionId}", Theme.Green));
        }
    }

    private static void Shop(GameState state, List<Line> lines)
    {
        if (state.Shop is not { } shop)
        {
            return;
        }
        lines.Add(new Line().Add("SHOP", Theme.Gold).Dim($"   ({shop.Gold}g available)"));
        foreach (ShopItemView i in shop.Items)
        {
            string name = i.ItemType switch
            {
                "Card" => Localizer.CardName(i.ItemId),
                "Relic" => Localizer.RelicName(i.ItemId),
                "Potion" => Localizer.PotionName(i.ItemId),
                _ => i.ItemId,
            };
            var l = new Line();
            l.Dim($"  {i.ItemType,-8} ").T($"{name,-24} ");
            l.Add($"{i.Cost}g", i.Affordable ? Theme.Gold : Theme.Dim);
            if (!i.Affordable)
            {
                l.Add("  (can't afford)", Theme.Red);
            }
            lines.Add(l);
        }
    }

    private static void CrystalSphere(GameState state, List<Line> lines)
    {
        if (state.CrystalSphere is not { } cs)
        {
            return;
        }
        var hidden = cs.HiddenCells.ToHashSet();
        lines.Add(new Line().Add("CRYSTAL SPHERE", Theme.Teal)
            .Dim($"   divinations {cs.DivinationsLeft} · tool {cs.Tool}"));
        for (int y = 0; y < cs.Height; y++)
        {
            var l = new Line().T("  ");
            for (int x = 0; x < cs.Width; x++)
            {
                bool h = hidden.Contains(new Coord(x, y));
                l.Add(h ? "▓ " : "· ", h ? Theme.Dim : Theme.Teal);
            }
            lines.Add(l);
        }
        foreach (CrystalSphereItemView it in cs.Items)
        {
            lines.Add(new Line()
                .Add(it.IsGood ? "  + " : "  - ", it.IsGood ? Theme.Green : Theme.Red)
                .T(it.ItemType + " ")
                .Add(it.Revealed ? "revealed" : "hidden", it.Revealed ? Theme.Green : Theme.Dim));
        }
    }

    private static void GameOver(GameState state, List<Line> lines)
    {
        lines.Add(state.IsVictory
            ? new Line().Add("V I C T O R Y", Theme.Green)
            : new Line().Add("D E F E A T", Theme.Red));
        lines.Add(new Line());
        lines.Add(new Line().T($"Reached Act {state.ActIndex + 1}, floor {state.Floor}."));
        lines.Add(new Line().T("Final score: ").Add(state.Score.ToString(), Theme.Gold));
        lines.Add(new Line().Dim($"Seed: {state.Seed}"));
        lines.Add(new Line());
        lines.Add(new Line().Dim("Game ▸ New Run to play again, or Game ▸ Quit."));
    }

    // ---- Map -------------------------------------------------------------------

    public static void MapLines(MapView map, List<Line> lines, int width, int height = 1000, Coord? highlight = null)
    {
        if (map.Points.Count == 0)
        {
            lines.Add(new Line().Dim("(no map)"));
            return;
        }

        // When a move option is highlighted, "lit" is that node plus everything reachable onward from
        // it; nodes outside the set are drawn dark so it's clear where that move would let you go.
        HashSet<Coord>? lit = highlight is { } hi ? Descendants(map, hi) : null;

        var header = new Line().Add($"Act {map.ActIndex + 1}", Theme.Gold);
        if (map.BossEncounterId is { } bossId)
        {
            header.Dim("  ").Add("Boss: ", Theme.Red).Add(Localizer.EncounterName(bossId), Theme.Fg);
            if (map.SecondBossEncounterId is { } secondBossId)
            {
                header.Dim(" + ").Add(Localizer.EncounterName(secondBossId), Theme.Fg);
            }
        }
        lines.Add(header);

        int maxCol = map.Points.Max(p => p.Coord.Col);
        int maxRow = map.Points.Max(p => p.Coord.Row);
        var reachable = map.Reachable.ToHashSet();
        const int cw = 3; // 3 columns per node: marker, icon, marker
        int gw = (maxCol + 1) * cw;
        int NodeCol(int c) => c * cw + 1;

        // The map is often taller than the panel; window it around where the player is so the
        // current/reachable rooms (which are what you act on) are always visible. Show from the focus
        // row (current, or the start) upward — the rooms ahead — for as many rows as fit.
        int focus = map.CurrentCoord?.Row ?? 0;
        int bodyRows = Math.Max(3, (height - 4) / 2); // header + legend ≈ 4 lines; 2 display lines/row
        int loRow = focus;
        int hiRow = Math.Min(maxRow, focus + bodyRows - 1);
        if (hiRow - loRow + 1 < bodyRows)
        {
            loRow = Math.Max(0, hiRow - bodyRows + 1); // fill upward if near the top
        }

        // Top (boss) to bottom (start). Between each node row, a connector row shows the edges from the
        // lower row's nodes up to their children in the row above.
        for (int row = hiRow; row >= loRow; row--)
        {
            var ch = new char[gw];
            var co = new Color[gw];
            for (int i = 0; i < gw; i++)
            {
                ch[i] = ' ';
                co[i] = Theme.Dim;
            }
            foreach (MapPointView pt in map.Points.Where(p => p.Coord.Row == row))
            {
                int nc = NodeCol(pt.Coord.Col);
                bool cur = map.CurrentCoord is { } cc && cc.Equals(pt.Coord);
                bool go = reachable.Contains(pt.Coord);
                bool isHi = highlight is { } hh && hh.Equals(pt.Coord);
                // Darken nodes the highlighted move can't lead to, but keep the "you are here" node lit
                // as an anchor.
                bool dark = lit is not null && !lit.Contains(pt.Coord) && !cur;
                char lb, rb, icon = Icon(pt.PointType)[0];
                Color marker, iconColor;
                if (dark)
                {
                    // Not reachable from the highlighted move: draw it dark/desaturated.
                    lb = rb = ' ';
                    marker = iconColor = Theme.HpLost;
                }
                else if (isHi)
                {
                    lb = '{'; rb = '}';
                    marker = iconColor = Theme.Gold;
                }
                else if (cur)
                {
                    lb = '['; rb = ']';
                    marker = iconColor = Theme.Fg;
                }
                else if (go)
                {
                    lb = '<'; rb = '>';
                    marker = iconColor = Theme.Green;
                }
                else
                {
                    lb = rb = ' ';
                    marker = Theme.Dim;
                    iconColor = IconColor(pt.PointType);
                }
                Put(ch, co, nc - 1, lb, marker);
                Put(ch, co, nc, icon, iconColor);
                Put(ch, co, nc + 1, rb, marker);
            }
            lines.Add(CellsToLine(ch, co));

            if (row > loRow)
            {
                var cch = new char[gw];
                var cco = new Color[gw];
                for (int i = 0; i < gw; i++)
                {
                    cch[i] = ' ';
                    cco[i] = Theme.Dim;
                }
                foreach (MapPointView pt in map.Points.Where(p => p.Coord.Row == row - 1))
                {
                    int nc = NodeCol(pt.Coord.Col);
                    foreach (Coord child in pt.Children.Where(c => c.Row == row))
                    {
                        // An edge is lit only when it stays inside the highlighted subtree (both ends lit).
                        bool onPath = lit is null || (lit.Contains(pt.Coord) && lit.Contains(child));
                        Color edge = onPath ? Theme.Dim : Theme.HpLost;
                        if (child.Col == pt.Coord.Col)
                        {
                            Put(cch, cco, nc, '|', edge);
                        }
                        else if (child.Col < pt.Coord.Col)
                        {
                            Put(cch, cco, nc - 1, '\\', edge);
                        }
                        else
                        {
                            Put(cch, cco, nc + 1, '/', edge);
                        }
                    }
                }
                lines.Add(CellsToLine(cch, cco));
            }
        }

        lines.Add(new Line());
        lines.Add(new Line()
            .Dim("M").T("onst ").Add("E", Theme.Magenta).T("lite ")
            .Add("B", Theme.Red).T("oss ").Add("R", Theme.Green).T("est"));
        lines.Add(new Line()
            .Add("$", Theme.Gold).T("hop ").Add("T", Theme.Gold).T("reas ")
            .Add("?", Theme.Teal).T("evt  ").Add("[x]", Theme.Fg).Dim("here ").Add("<x>", Theme.Green).Dim("go ")
            .Add("{x}", Theme.Gold).Dim("sel"));
    }

    // The highlighted node plus every node reachable onward from it (its descendants in the map graph),
    // so the map can dim everything a given move can't lead to.
    private static HashSet<Coord> Descendants(MapView map, Coord start)
    {
        var children = map.Points.ToDictionary(p => p.Coord, p => p.Children);
        var seen = new HashSet<Coord> { start };
        var stack = new Stack<Coord>();
        stack.Push(start);
        while (stack.Count > 0)
        {
            Coord c = stack.Pop();
            if (children.TryGetValue(c, out IReadOnlyList<Coord>? kids))
            {
                foreach (Coord k in kids)
                {
                    if (seen.Add(k))
                    {
                        stack.Push(k);
                    }
                }
            }
        }
        return seen;
    }

    private static void Put(char[] ch, Color[] co, int i, char c, Color color)
    {
        if (i >= 0 && i < ch.Length)
        {
            ch[i] = c;
            co[i] = color;
        }
    }

    private static Line CellsToLine(char[] ch, Color[] co)
    {
        var l = new Line();
        var sb = new System.Text.StringBuilder();
        Color? run = null;
        for (int i = 0; i < ch.Length; i++)
        {
            if (run != co[i])
            {
                if (sb.Length > 0 && run.HasValue)
                {
                    l.Add(sb.ToString(), run.Value);
                    sb.Clear();
                }
                run = co[i];
            }
            sb.Append(ch[i]);
        }
        if (sb.Length > 0 && run.HasValue)
        {
            l.Add(sb.ToString(), run.Value);
        }
        return l;
    }

    private static string Icon(MapPointType t) => t switch
    {
        MapPointType.Monster => "M",
        MapPointType.Elite => "E",
        MapPointType.Boss => "B",
        MapPointType.RestSite => "R",
        MapPointType.Shop => "$",
        MapPointType.Treasure => "T",
        MapPointType.Ancient => "A",
        _ => "?",
    };

    private static Color IconColor(MapPointType t) => t switch
    {
        MapPointType.Monster => Theme.Dim,
        MapPointType.Elite => Theme.Magenta,
        MapPointType.Boss => Theme.Red,
        MapPointType.RestSite => Theme.Green,
        MapPointType.Shop => Theme.Gold,
        MapPointType.Treasure => Theme.Gold,
        _ => Theme.Teal,
    };

    // ---- Deck popup ------------------------------------------------------------

    public static List<Line> Deck(PlayerState p)
    {
        var lines = new List<Line>
        {
            new Line().Add(p.Character, Theme.Green)
                .Dim("  ").Add($"{p.CurrentHp}/{p.MaxHp} HP", Theme.Hp(p.CurrentHp, p.MaxHp))
                .Dim("  ").Add($"{p.Gold}g", Theme.Gold),
            new Line(),
            new Line().Add($"DECK ({p.Deck.Count})", Theme.Gold),
        };
        foreach (var grp in p.Deck.GroupBy(c => (CardName(c), ModifierKey(c))).OrderBy(g => g.Key.Item1))
        {
            var l = new Line().T($"  {grp.Key.Item1} ");
            AppendModifierSegs(l, grp.First());
            l.Dim($" x{grp.Count()}");
            lines.Add(l);
        }
        lines.Add(new Line());
        lines.Add(new Line().Add($"RELICS ({p.Relics.Count})", Theme.Teal));
        lines.Add(new Line().T("  " + (p.Relics.Count == 0 ? "(none)" : string.Join(", ", p.Relics.Select(Localizer.RelicName)))));
        lines.Add(new Line());
        lines.Add(new Line().Add("POTIONS", Theme.Teal));
        lines.Add(new Line().T("  " + string.Join("   ", p.Potions.Select(x => x is null ? "(empty)" : Localizer.PotionName(x)))));
        return lines;
    }

    // ---- Option labels & descriptions (the decision area) ----------------------

    /// <summary>The coloured header label for an option (energy cost as teal circles for cards).</summary>
    public static List<Seg> OptionLabel(GameOption o, GameState state) => o.Kind switch
    {
        OptionKind.PlayCard when o.Card is { } c => CardLabelSegs(c, o.TargetCombatId),
        OptionKind.SelectCards when (o.SelectedCards?.Count ?? 0) == 0 => Text("Skip selection", Theme.Dim),
        OptionKind.SelectCards when o.Card is { } c => CardLabelSegs(c, null),
        OptionKind.MoveTo => Text(MoveLabel(o, state)),
        OptionKind.EndTurn => Text("End turn", Theme.Gold),
        OptionKind.ProceedFromRewards => Text("Proceed (leave screen)"),
        OptionKind.SkipTreasure => Text("Skip — take no relic", Theme.Dim),
        OptionKind.ChooseEventOption => EventLabelSegs(o, state),
        OptionKind.TakeReward when o.Card is { } rc => Prepend("Take ", CardLabelSegs(rc, null)),
        OptionKind.TakeTreasureRelic when o.TreasureRelicId is { } rid => Text($"Take {Localizer.RelicName(rid)}"),
        OptionKind.BuyShopItem => ShopLabelSegs(o),
        OptionKind.UsePotion when o.PotionId is { } pid => Text(
            $"Use {Localizer.PotionName(pid)}" + (o.TargetCombatId is { } id ? $" → #{id}" : "")),
        OptionKind.DiscardPotion when o.PotionId is { } pid => Text($"Discard {Localizer.PotionName(pid)}", Theme.Dim),
        _ => Text(o.Description),
    };

    /// <summary>The option's localized description parsed into coloured segments (energy → teal circles).</summary>
    public static List<Seg> OptionDescSegs(GameOption o, GameState state) =>
        Markup.Parse(Subject(o, state).desc, Theme.Fg);

    private static List<Seg> Text(string s, Color? c = null) => new() { new(s, c ?? Theme.Fg) };

    private static List<Seg> Prepend(string pre, List<Seg> segs)
    {
        segs.Insert(0, new Seg(pre, Theme.Fg));
        return segs;
    }

    /// <summary>A card rendered as coloured segments (cost circle, name, damage/block preview,
    /// modifiers) — used by the interactive card-selection picker.</summary>
    public static List<Seg> CardSegs(CardView c) => CardLabelSegs(c, null);

    /// <summary>The card's display name (with a trailing "+" when upgraded).</summary>
    public static string CardDisplayName(CardView c) => CardName(c);

    /// <summary>The card's localized rules text.</summary>
    public static string CardDescription(CardView c) => Localizer.CardDescription(c.CardId, c.Upgraded);

    private static List<Seg> CardLabelSegs(CardView c, uint? target)
    {
        var segs = new List<Seg> { Markup.Cost(c.CostsX, c.EnergyCost), new(" ", Theme.Fg), new(CardName(c), Theme.Fg) };
        if (target is { } id)
        {
            segs.Add(new Seg($"  → #{id}", Theme.Dim));
        }
        AppendEffectSegs(segs, c);
        AppendModifierSegs(segs, c);
        return segs;
    }

    private static List<Seg> ShopLabelSegs(GameOption o)
    {
        var segs = new List<Seg>();
        if (o.ShopItemType == "Card" && o.Card is { } c)
        {
            segs.Add(Markup.Cost(c.CostsX, c.EnergyCost));
            segs.Add(new Seg(" ", Theme.Fg));
            segs.Add(new Seg(CardName(c), Theme.Fg));
        }
        else
        {
            string name = o.ShopItemType switch
            {
                "Relic" => Localizer.RelicName(o.ShopItemId ?? "?"),
                "Potion" => Localizer.PotionName(o.ShopItemId ?? "?"),
                "CardRemoval" => "Card removal",
                _ => o.ShopItemId ?? "?",
            };
            segs.Add(new Seg(name, Theme.Fg));
        }
        segs.Add(new Seg($"  {o.ShopItemCost}g", Theme.Gold));
        return segs;
    }

    private static List<Seg> EventLabelSegs(GameOption o, GameState state)
    {
        EventOptionView? ov = state.Event?.Options.FirstOrDefault(x => x.Index == o.EventOptionIndex);
        string eventId = state.Event?.EventId ?? "";
        string title = EventTitle(eventId, ov, ov?.TextKey, o.EventOptionRelicId);
        var segs = new List<Seg> { new(title, Theme.Fg) };
        if (o.EventOptionRelicId is { } rid && !title.Contains(Localizer.RelicName(rid)))
        {
            segs.Add(new Seg($"  (relic {Localizer.RelicName(rid)})", Theme.Teal));
        }
        return segs;
    }

    /// <summary>
    /// An event option's display title: the live rendered title (with correct dynamic numbers, markup
    /// stripped), else the by-key localized title, else the granted relic's name, else a prettified
    /// last segment of the text key.
    /// </summary>
    public static string EventTitle(string eventId, EventOptionView? ov, string? textKey, string? relicId)
    {
        if (ov?.Title is { } live && Localizer.Clean(live) is { Length: > 0 } cleaned)
        {
            return cleaned;
        }
        if (textKey is not null && Localizer.EventOptionTitle(eventId, textKey) is { } t)
        {
            return t;
        }
        if (relicId is not null)
        {
            return Localizer.RelicName(relicId);
        }
        return Pretty(textKey);
    }

    /// <summary>
    /// An event option's outcome text (raw markup kept for colouring): the live rendered description
    /// (correct dynamic numbers), else the by-key localized description, else the granted relic's.
    /// </summary>
    public static string EventDesc(string eventId, EventOptionView? ov, string? textKey, string? relicId)
    {
        if (ov?.Description is { } live && !string.IsNullOrWhiteSpace(live))
        {
            return live;
        }
        if (textKey is not null && Localizer.EventOptionDescription(eventId, textKey) is { } d)
        {
            return d;
        }
        return relicId is not null ? Localizer.RelicDescription(relicId) : string.Empty;
    }

    private static string Pretty(string? key)
    {
        if (string.IsNullOrEmpty(key))
        {
            return "(option)";
        }
        string seg = key.Split('.').Last().Replace('_', ' ').ToLowerInvariant();
        return System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase(seg);
    }

    private static string MoveLabel(GameOption o, GameState state)
    {
        Coord coord = o.Coord!.Value;
        string type = state.Map?.Points.FirstOrDefault(p => p.Coord.Equals(coord))?.PointType.ToString() ?? "room";
        return $"Travel to {type} ({coord.Col},{coord.Row})";
    }

    // ---- Per-option localized description (shown under each option in the decision area) ---------

    private static (string name, string desc) Subject(GameOption o, GameState state) => o.Kind switch
    {
        OptionKind.PlayCard when o.Card is { } c => (CardName(c), Localizer.CardDescription(c.CardId, c.Upgraded)),
        OptionKind.SelectCards when o.Card is { } c => (CardName(c), Localizer.CardDescription(c.CardId, c.Upgraded)),
        OptionKind.TakeReward when o.Card is { } c => (CardName(c), Localizer.CardDescription(c.CardId, c.Upgraded)),
        OptionKind.TakeTreasureRelic when o.TreasureRelicId is { } rid => (Localizer.RelicName(rid), Localizer.RelicDescription(rid)),
        OptionKind.BuyShopItem => ShopSubject(o),
        OptionKind.UsePotion when o.PotionId is { } pid => (Localizer.PotionName(pid), Localizer.PotionDescription(pid)),
        // Discard shows no description: the potion's use-text ("gain regen 5") reads as if discarding
        // grants the effect, which is confusing. The label already names the potion being discarded.
        OptionKind.DiscardPotion when o.PotionId is { } pid => (Localizer.PotionName(pid), string.Empty),
        OptionKind.ChooseEventOption => EventSubject(o, state),
        _ => (string.Empty, string.Empty),
    };

    private static (string, string) ShopSubject(GameOption o) => o.ShopItemType switch
    {
        "Card" when o.Card is { } c => (CardName(c), Localizer.CardDescription(c.CardId, c.Upgraded)),
        "Relic" => (Localizer.RelicName(o.ShopItemId ?? "?"), Localizer.RelicDescription(o.ShopItemId ?? "?")),
        "Potion" => (Localizer.PotionName(o.ShopItemId ?? "?"), Localizer.PotionDescription(o.ShopItemId ?? "?")),
        "CardRemoval" => ("Card Removal", "Remove a card from your deck."),
        _ => (o.ShopItemId ?? "?", string.Empty),
    };

    private static (string, string) EventSubject(GameOption o, GameState state)
    {
        EventOptionView? ov = state.Event?.Options.FirstOrDefault(x => x.Index == o.EventOptionIndex);
        string eventId = state.Event?.EventId ?? "";
        return (EventTitle(eventId, ov, ov?.TextKey, o.EventOptionRelicId),
                EventDesc(eventId, ov, ov?.TextKey, o.EventOptionRelicId));
    }

}
