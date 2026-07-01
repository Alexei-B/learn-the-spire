using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Runs;

namespace Lts2.Harness;

/// <summary>
/// Projects the live game-logic singletons into the immutable <see cref="GameState"/>
/// read model. All access is read-only; nothing here mutates game state.
/// </summary>
internal static class GameStateProjection
{
    public static GameState Capture(GameHost host)
    {
        RunState run = host.Run;
        CombatState? combat = host.Combat;
        PendingChoice? pending = host.Selector.Pending;
        GamePhase phase = DeterminePhase(host, run, pending);

        return new GameState
        {
            Phase = phase,
            Seed = host.Seed,
            ActIndex = run.CurrentActIndex,
            Floor = run.TotalFloor,
            AscensionLevel = run.AscensionLevel,
            IsGameOver = run.IsGameOver,
            IsVictory = run.IsGameOver && RunManager.Instance.WinTime > 0,
            Score = ScoreUtility.CalculateScore(run, won: RunManager.Instance.WinTime > 0),
            Players = run.Players.Select(p => ProjectPlayer(p, combat)).ToList(),
            Combat = combat is null ? null : ProjectCombat(combat),
            Map = ProjectMap(run),
            PendingChoice = pending is null ? null : ProjectPendingChoice(pending),
            Rewards = host.PendingRewards is null ? null : ProjectRewards(host.PendingRewards),
            Event = host.HasActionableEvent ? ProjectEvent(host.CurrentEvent!, run) : null,
            Treasure = host.HasTreasureChoice ? ProjectTreasure(host) : null,
            RestSite = host.HasRestChoice ? ProjectRestSite() : null,
            Shop = host.HasShopChoice ? ProjectShop(host) : null,
            CrystalSphere = host.PendingCrystalSphere is { } mg ? ProjectCrystalSphere(mg) : null,
        };
    }

    private static GamePhase DeterminePhase(GameHost host, RunState run, PendingChoice? pending)
    {
        if (!RunManager.Instance.IsInProgress)
        {
            return GamePhase.NotStarted;
        }
        if (run.IsGameOver)
        {
            return GamePhase.GameOver;
        }
        if (pending is not null)
        {
            return GamePhase.Choice;
        }
        if (host.PendingCrystalSphere is not null)
        {
            return GamePhase.CrystalSphere;
        }
        if (host.InCombat)
        {
            return GamePhase.Combat;
        }
        if (host.PendingRewards is not null)
        {
            return GamePhase.Reward;
        }
        if (host.HasTreasureChoice)
        {
            return GamePhase.Treasure;
        }
        if (host.HasRestChoice)
        {
            return GamePhase.RestSite;
        }
        if (host.HasShopChoice)
        {
            return GamePhase.Shop;
        }
        if (host.HasActionableEvent)
        {
            return GamePhase.Event;
        }
        // No room yet, or sitting on a map point we can move off of: treat as Map when
        // there is somewhere to go, otherwise as an unmodelled room/screen.
        return ReachablePoints(run).Any() ? GamePhase.Map : GamePhase.Other;
    }

    private static PlayerState ProjectPlayer(Player player, CombatState? combat)
    {
        PlayerCombatState? pcs = combat is null ? null : player.PlayerCombatState;
        return new PlayerState
        {
            NetId = player.NetId,
            Character = player.Character.Id.Entry,
            CurrentHp = player.Creature.CurrentHp,
            MaxHp = player.Creature.MaxHp,
            Block = player.Creature.Block,
            Gold = player.Gold,
            MaxEnergy = player.MaxEnergy,
            Deck = player.Deck.Cards.Select(c => ProjectCard(c, canPlay: false)).ToList(),
            Relics = player.Relics.Select(r => r.Id.Entry).ToList(),
            Potions = player.PotionSlots.Select(p => p?.Id.Entry).ToList(),
            CombatState = pcs is null ? null : ProjectPlayerCombat(player, pcs),
        };
    }

    private static PlayerCombatView ProjectPlayerCombat(Player player, PlayerCombatState pcs) =>
        new()
        {
            Energy = pcs.Energy,
            MaxEnergy = pcs.MaxEnergy,
            Stars = pcs.Stars,
            TurnNumber = pcs.TurnNumber,
            Phase = pcs.Phase,
            Hand = pcs.Hand.Cards.Select(c => ProjectCard(c, c.CanPlay())).ToList(),
            DrawPile = pcs.DrawPile.Cards.Select(c => ProjectCard(c, canPlay: false)).ToList(),
            DiscardPile = pcs.DiscardPile.Cards.Select(c => ProjectCard(c, canPlay: false)).ToList(),
            ExhaustPile = pcs.ExhaustPile.Cards.Select(c => ProjectCard(c, canPlay: false)).ToList(),
            Powers = player.Creature.Powers.Select(ProjectPower).ToList(),
            Orbs = pcs.OrbQueue.Orbs.Select(ProjectOrb).ToList(),
            OrbSlots = pcs.OrbQueue.Capacity,
            Osty = player.Osty is { } osty ? ProjectOsty(osty) : null,
        };

    private static OrbView ProjectOrb(MegaCrit.Sts2.Core.Models.OrbModel orb) =>
        new()
        {
            OrbId = orb.Id.Entry,
            PassiveValue = (int)orb.PassiveVal,
            EvokeValue = (int)orb.EvokeVal,
        };

    private static OstyView ProjectOsty(Creature osty) =>
        new()
        {
            CurrentHp = osty.CurrentHp,
            MaxHp = osty.MaxHp,
            Block = osty.Block,
            IsAlive = osty.IsAlive,
            Powers = osty.Powers.Select(ProjectPower).ToList(),
        };

    internal static CardView ProjectCard(
        CardModel card, bool canPlay, bool? upgradedOverride = null, Creature? target = null)
    {
        CostModifiers modifiers = card.IsInCombat ? CostModifiers.All : CostModifiers.None;
        (int? dmg, int? baseDmg, int? block, int? baseBlock) = CardEffectPreview(card, target);
        return new CardView
        {
            CardId = card.Id.Entry,
            EnergyCost = card.EnergyCost.GetWithModifiers(modifiers),
            CostsX = card.EnergyCost.CostsX,
            Type = card.Type,
            Rarity = card.Rarity,
            TargetType = card.TargetType,
            // The forge shows each candidate as the upgraded card it would become; override the flag so
            // the UI renders it with a "+" and the upgraded description.
            Upgraded = upgradedOverride ?? card.IsUpgraded,
            CanPlay = canPlay,
            Damage = dmg,
            BaseDamage = baseDmg,
            Block = block,
            BaseBlock = baseBlock,
            StarCost = StarCostOf(card),
        };
    }

    /// <summary>The card's star cost (Regent's second resource), or -1 for cards that don't use stars.
    /// Guarded because the X-star path dereferences the owner's combat state.</summary>
    private static int StarCostOf(CardModel card)
    {
        try
        {
            return card.GetStarCostWithModifiers();
        }
        catch
        {
            return card.CanonicalStarCost;
        }
    }

    /// <summary>
    /// The actual damage/block a card would produce right now, plus the unmodified base values (so the
    /// UI can colour buffs green and debuffs red). Uses the game's own preview path
    /// (<c>UpdateDynamicVarPreview</c> → <c>Hook.ModifyDamage/ModifyBlock</c>), which folds in the
    /// attacker's powers (Strength/Weak/Vigor/Sovereign Blade/…) and, when a <paramref name="target"/>
    /// is given, that defender's powers (Vulnerable/Intangible/…). Only meaningful for a hand card in
    /// combat; returns nulls otherwise (or if the preview throws, as some cards compute lazily).
    /// </summary>
    private static (int? dmg, int? baseDmg, int? block, int? baseBlock) CardEffectPreview(
        CardModel card, Creature? target)
    {
        if (!card.IsInCombat)
        {
            return (null, null, null, null);
        }
        try
        {
            card.UpdateDynamicVarPreview(
                MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode.Normal, target, card.DynamicVars);
            int? dmg = null, baseDmg = null, block = null, baseBlock = null;
            if (card.DynamicVars.ContainsKey("Damage"))
            {
                var d = card.DynamicVars.Damage;
                dmg = Math.Max(0, (int)d.PreviewValue);
                baseDmg = Math.Max(0, (int)d.BaseValue);
            }
            if (card.DynamicVars.ContainsKey("Block"))
            {
                var b = card.DynamicVars.Block;
                block = Math.Max(0, (int)b.PreviewValue);
                baseBlock = Math.Max(0, (int)b.BaseValue);
            }
            return (dmg, baseDmg, block, baseBlock);
        }
        catch
        {
            return (null, null, null, null);
        }
    }

    private static PendingChoiceView ProjectPendingChoice(PendingChoice pending)
    {
        bool? upgradePreview = pending.IsUpgradeSelection ? true : (bool?)null;
        return new PendingChoiceView
        {
            Options = pending.Options.Select(c => ProjectCard(c, canPlay: false, upgradePreview)).ToList(),
            MinSelect = pending.MinSelect,
            MaxSelect = pending.MaxSelect,
            IsUpgradeSelection = pending.IsUpgradeSelection,
        };
    }

    private static EventView ProjectEvent(MegaCrit.Sts2.Core.Models.EventModel ev, RunState run)
    {
        var options = new List<EventOptionView>();
        IReadOnlyList<MegaCrit.Sts2.Core.Events.EventOption> current = ev.CurrentOptions;
        for (int i = 0; i < current.Count; i++)
        {
            MegaCrit.Sts2.Core.Events.EventOption opt = current[i];
            if (opt.IsLocked || opt.IsProceed)
            {
                continue;
            }
            options.Add(new EventOptionView
            {
                Index = i,
                TextKey = opt.TextKey,
                Title = RenderEventLoc(ev, opt.Title),
                Description = RenderEventLoc(ev, opt.Description),
                RelicId = opt.Relic?.Id.Entry,
            });
        }

        // For a shared (vote-based) event, surface each player's pending vote so an agent can see what
        // the others have indicated. Per-player events resolve independently — no votes to show.
        var votes = new List<EventVoteView>();
        if (ev.IsShared)
        {
            MegaCrit.Sts2.Core.Multiplayer.Game.EventSynchronizer sync = RunManager.Instance.EventSynchronizer;
            foreach (Player p in run.Players)
            {
                uint? vote = sync.GetPlayerVote(p);
                votes.Add(new EventVoteView
                {
                    NetId = p.NetId,
                    HasVoted = vote.HasValue,
                    VotedOptionIndex = vote.HasValue ? (int)vote.Value : null,
                });
            }
        }

        return new EventView
        {
            EventId = ev.Id.Entry,
            Description = RenderEventLoc(ev, ev.Description) ?? RenderEventLoc(ev, SafeInitialDescription(ev)),
            IsAncient = ev is MegaCrit.Sts2.Core.Models.AncientEventModel,
            IsShared = ev.IsShared,
            Options = options,
            Votes = votes,
        };
    }

    /// <summary>The event's initial-page description LocString, or null if it cannot be resolved.</summary>
    private static MegaCrit.Sts2.Core.Localization.LocString? SafeInitialDescription(
        MegaCrit.Sts2.Core.Models.EventModel ev)
    {
        try
        {
            return ev.InitialDescription;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Render an event LocString after binding the event's dynamic variables (Enchantment names,
    /// amounts, …) — the same step the game's event UI does before formatting
    /// (<c>Event.DynamicVars.AddTo(...)</c>). Without it, per-run placeholders like
    /// <c>{Enchantment1Amount}</c> stay unbound and formatting fails. Returns null on any failure.
    /// </summary>
    private static string? RenderEventLoc(
        MegaCrit.Sts2.Core.Models.EventModel ev, MegaCrit.Sts2.Core.Localization.LocString? ls)
    {
        if (ls is null)
        {
            return null;
        }
        try
        {
            ev.DynamicVars.AddTo(ls);
        }
        catch
        {
            // Fall through and render whatever binds — RenderLoc still guards the format call.
        }
        return RenderLoc(ls);
    }

    /// <summary>
    /// Render a live game LocString to raw markup text (dynamic numbers already bound), or null when it
    /// is null, absent, whitespace, or throws. Kept id-agnostic: the harness stays pak-free (missing
    /// localization degrades to the key via the Harmony patch); callers may parse/strip the markup.
    /// </summary>
    private static string? RenderLoc(MegaCrit.Sts2.Core.Localization.LocString? ls)
    {
        try
        {
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

    private static TreasureView ProjectTreasure(GameHost host)
    {
        MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer sync =
            RunManager.Instance.TreasureRoomRelicSynchronizer;
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Models.RelicModel>? relics =
            host.CurrentTreasureRoom is null ? null : sync.CurrentRelics;

        // In a multi-player chest, surface each player's pending pick so an agent sees the others'
        // choices before it resolves. Single-player has no votes to show.
        var votes = new List<TreasureVoteView>();
        if (relics is not null && host.Run.Players.Count > 1)
        {
            foreach (Player p in host.Run.Players)
            {
                MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer.PlayerVote vote =
                    sync.GetPlayerVote(p);
                votes.Add(new TreasureVoteView
                {
                    NetId = p.NetId,
                    HasVoted = vote.voteReceived,
                    VotedRelicIndex = vote.voteReceived ? vote.index : null,
                });
            }
        }

        return new TreasureView
        {
            Relics = relics is null
                ? Array.Empty<string>()
                : relics.Select(r => r.Id.Entry).ToList(),
            Votes = votes,
        };
    }

    private static CrystalSphereView ProjectCrystalSphere(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame)
    {
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell[,] cells = minigame.cells;
        int width = cells.GetLength(0);
        int height = cells.GetLength(1);

        var hidden = new List<Coord>();
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (cells[x, y].IsHidden)
                {
                    hidden.Add(new Coord(x, y));
                }
            }
        }

        var items = minigame.Items.Select(item => new CrystalSphereItemView
        {
            ItemType = item.GetType().Name.Replace("CrystalSphere", string.Empty),
            IsGood = item.IsGood,
            Position = new Coord(item.Position.X, item.Position.Y),
            Size = new Coord(item.Size.X, item.Size.Y),
            Revealed = IsItemFullyRevealed(cells, item),
        }).ToList();

        return new CrystalSphereView
        {
            Width = width,
            Height = height,
            DivinationsLeft = minigame.DivinationCount,
            Tool = minigame.CrystalSphereTool.ToString(),
            HiddenCells = hidden,
            Items = items,
        };
    }

    private static bool IsItemFullyRevealed(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell[,] cells,
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItem item)
    {
        for (int i = 0; i < item.Size.X; i++)
        {
            for (int j = 0; j < item.Size.Y; j++)
            {
                if (cells[item.Position.X + i, item.Position.Y + j].IsHidden)
                {
                    return false;
                }
            }
        }
        return true;
    }

    private static ShopView ProjectShop(GameHost host)
    {
        MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory inv = host.CurrentMerchantRoom!.GetLocalInventory();
        var items = new List<ShopItemView>();
        foreach (MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry in inv.AllEntries)
        {
            if (!entry.IsStocked)
            {
                continue;
            }
            (string type, string id, CardModel? card) = GameHost.ClassifyShopEntry(entry);
            items.Add(new ShopItemView
            {
                ItemType = type,
                ItemId = id,
                Cost = entry.Cost,
                Affordable = entry.EnoughGold,
                Card = card is null ? null : ProjectCard(card, canPlay: false),
            });
        }
        return new ShopView { Gold = host.Run.Players[0].Gold, Items = items };
    }

    private static RestSiteView ProjectRestSite()
    {
        var options = new List<RestSiteOptionView>();
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption> rest =
            RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
        for (int i = 0; i < rest.Count; i++)
        {
            if (rest[i].IsEnabled)
            {
                options.Add(new RestSiteOptionView { Index = i, OptionId = rest[i].OptionId });
            }
        }
        return new RestSiteView { Options = options };
    }

    private static RewardsView ProjectRewards(MegaCrit.Sts2.Core.Rewards.RewardsSet set) =>
        new() { Rewards = set.Rewards.Select(ProjectReward).ToList() };

    private static RewardView ProjectReward(MegaCrit.Sts2.Core.Rewards.Reward reward) => reward switch
    {
        MegaCrit.Sts2.Core.Rewards.GoldReward gold => new RewardView
        {
            Type = MegaCrit.Sts2.Core.Rewards.RewardType.Gold,
            Taken = gold.SuccessfullySelected,
            Gold = gold.Amount,
        },
        MegaCrit.Sts2.Core.Rewards.PotionReward potion => new RewardView
        {
            Type = MegaCrit.Sts2.Core.Rewards.RewardType.Potion,
            Taken = potion.SuccessfullySelected,
            PotionId = potion.Potion?.Id.Entry,
        },
        MegaCrit.Sts2.Core.Rewards.RelicReward relic => new RewardView
        {
            Type = MegaCrit.Sts2.Core.Rewards.RewardType.Relic,
            Taken = relic.SuccessfullySelected,
            RelicId = relic.Relic?.Id.Entry,
        },
        MegaCrit.Sts2.Core.Rewards.CardReward card => new RewardView
        {
            Type = MegaCrit.Sts2.Core.Rewards.RewardType.Card,
            Taken = card.SuccessfullySelected,
            Cards = card.Cards.Select(c => ProjectCard(c, canPlay: false)).ToList(),
            CardAlternatives = MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative
                .Generate(card).Select(a => a.OptionId).Where(id => id != "Skip").ToList(),
        },
        _ => new RewardView { Type = MegaCrit.Sts2.Core.Rewards.RewardType.None, Taken = reward.SuccessfullySelected },
    };

    private static PowerView ProjectPower(PowerModel power) =>
        new() { PowerId = power.Id.Entry, Amount = power.Amount };

    private static CombatView ProjectCombat(CombatState combat) =>
        new()
        {
            RoundNumber = combat.RoundNumber,
            CurrentSide = combat.CurrentSide,
            Enemies = combat.Enemies.Select(e => ProjectEnemy(e, combat)).ToList(),
        };

    private static EnemyView ProjectEnemy(Creature enemy, CombatState combat) =>
        new()
        {
            CombatId = enemy.CombatId ?? 0u,
            MonsterId = enemy.Monster?.Id.Entry ?? "?",
            CurrentHp = enemy.CurrentHp,
            MaxHp = enemy.MaxHp,
            Block = enemy.Block,
            IsHittable = enemy.IsHittable,
            Powers = enemy.Powers.Select(ProjectPower).ToList(),
            Intents = ProjectIntents(enemy, combat),
        };

    private static IReadOnlyList<IntentView> ProjectIntents(Creature enemy, CombatState combat)
    {
        if (enemy.Monster is null)
        {
            return Array.Empty<IntentView>();
        }

        var result = new List<IntentView>();
        foreach (AbstractIntent intent in enemy.Monster.NextMove.Intents)
        {
            int? damage = null;
            int? baseDamage = null;
            int? hits = null;
            if (intent is AttackIntent attack)
            {
                // Damage depends on the defending side; the player creatures are the targets.
                try
                {
                    damage = attack.GetSingleDamage(combat.Allies, enemy);
                    // The unmodified base (before the enemy's Strength/Weak and the player's
                    // Vulnerable/etc.) so the UI can colour the difference.
                    if (attack.DamageCalc is { } calc)
                    {
                        baseDamage = Math.Max(0, (int)calc());
                    }
                    // Repeats is the total number of hits (SingleAttackIntent => 1,
                    // MultiAttackIntent => its repeat count; GetTotalDamage = single × Repeats).
                    hits = attack.Repeats;
                }
                catch
                {
                    // Some intents resolve damage lazily and may not be computable here.
                }
            }
            result.Add(new IntentView { Type = intent.IntentType, Damage = damage, BaseDamage = baseDamage, Hits = hits });
        }
        return result;
    }

    private static MapView? ProjectMap(RunState run)
    {
        ActMap map = run.Map;
        if (map is NullActMap)
        {
            return null;
        }

        // The starting node and the boss node(s) live outside the Grid (they are separate ActMap
        // properties), so GetAllMapPoints() omits them. Append them — deduping by coord — so the map
        // shows the act's opening Ancient at the bottom and the boss room at the top, wired up by the
        // existing connectors (the starting node lists row 1 as children; the top grid row lists the boss).
        var mapPoints = map.GetAllMapPoints().ToList();
        var seen = mapPoints.Select(p => p.coord).ToHashSet();
        void AddPoint(MapPoint p)
        {
            if (seen.Add(p.coord))
            {
                mapPoints.Add(p);
            }
        }
        AddPoint(map.StartingMapPoint);
        AddPoint(map.BossMapPoint);
        if (map.SecondBossMapPoint is { } secondBoss)
        {
            AddPoint(secondBoss);
        }

        var points = mapPoints
            .Select(p => new MapPointView
            {
                Coord = Coord.From(p.coord),
                PointType = p.PointType,
                Children = p.Children.Select(c => Coord.From(c.coord)).ToList(),
            })
            .ToList();

        ActModel act = run.Act;
        return new MapView
        {
            ActIndex = run.CurrentActIndex,
            CurrentCoord = run.CurrentMapCoord is { } cc ? Coord.From(cc) : null,
            Points = points,
            Reachable = ReachablePoints(run).Select(p => Coord.From(p.coord)).ToList(),
            BossEncounterId = act.BossEncounter?.Id.Entry,
            SecondBossEncounterId = act.SecondBossEncounter?.Id.Entry,
        };
    }

    /// <summary>
    /// The map points the player may move to next: the children of the current point, or — at the
    /// very start of an act, before any room has been entered — the act's starting node itself.
    /// Mirrors <c>NMapScreen.RecalculateTravelability</c>: with nothing visited yet, only the starting
    /// node (which is each act's opening Ancient — Neow in Act 1) is travelable, so the act's ancient
    /// reward is always the first stop. <see cref="RunState.CurrentMapPoint"/> is null exactly when
    /// nothing has been visited this act (it derives from the visited coords).
    /// </summary>
    internal static IEnumerable<MapPoint> ReachablePoints(RunState run)
    {
        if (run.Map is NullActMap)
        {
            return Array.Empty<MapPoint>();
        }
        MapPoint? current = run.CurrentMapPoint;
        IEnumerable<MapPoint> next = current is not null
            ? current.Children
            : new[] { run.Map.StartingMapPoint };
        return next.OrderBy(p => p.coord.col);
    }
}
