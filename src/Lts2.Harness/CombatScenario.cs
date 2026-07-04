using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Lts2.Harness;

/// <summary>
/// Builds an isolated, randomized <b>combat scenario</b> for agent training: a random character with a
/// random 15-card deck from its pool, its starting relic plus 5 random relics, at full HP, dropped
/// directly into a random act-1/2/3 encounter (weighted toward normals, with configurable elite/boss
/// rates). Unlike a full run, the episode is a single fight — the agent optimizes combat itself over a
/// far wider spread of decks/relics/enemies than a normal playthrough visits.
///
/// <para>Uses the same dev seams as the fight tests (<see cref="GameHost.EnterEncounterDebug"/>,
/// <see cref="GameHost.ObtainRelicDebug"/>): start a run for the character, resolve the opening event to
/// reach the map (a valid state for direct room entry), normalize the player (custom deck with each
/// card's <c>Owner</c> set — required or the run's hook iteration NREs on entry — starter-only relics,
/// full HP), grant the random relics, then enter the encounter.</para>
/// </summary>
public static class CombatScenario
{
    /// <summary>What was rolled for a scenario, plus the HP bookkeeping a reward needs.</summary>
    public sealed record Spec(
        string Character,
        string Encounter,
        int Act,
        string RoomType,
        int StartHp,
        int StarterHeal);

    /// <summary>
    /// The HP a character's <em>starting relic</em> restores at end of combat, which must be added back
    /// before scoring HP lost (so the reward reflects real combat damage, not the post-combat recovery).
    /// Currently only the Ironclad's Burning Blood (6). Others have no such starter heal (0).
    /// </summary>
    public static int StarterHealFor(string characterId) => characterId == "IRONCLAD" ? 6 : 0;

    private const int DeckSize = 15;
    private const int RelicCount = 5;

    /// <summary>
    /// Create a fresh combat scenario. Tears down any prior run (one run per process) and returns the
    /// host sitting in the built combat, plus the <see cref="Spec"/> describing it.
    /// </summary>
    /// <param name="characterName">Substring match against a character id; null/empty = random.</param>
    /// <param name="elitePct">Probability the encounter is an elite.</param>
    /// <param name="bossPct">Probability the encounter is a boss (the rest are normal monster fights).</param>
    /// <param name="useStarterDeck">Use the character's fixed starting deck + starting relic only
    /// (no random deck/relics) — a low-noise regime for focused training on one character's basics.</param>
    /// <param name="act">Restrict the random encounter to this act index (0/1/2); -1 = any.</param>
    public static (GameHost Host, Spec Spec) Create(
        string seed, Random rng, string? characterName, double elitePct, double bossPct,
        bool useStarterDeck = false, int act = -1)
    {
        long t = System.Diagnostics.Stopwatch.GetTimestamp();
        CharacterModel character = PickCharacter(rng, characterName);

        // Combat training never navigates the map (it drops straight into one encounter), so skip the
        // ~100ms path-segment pruning during map generation. Scoped to just the run setup so it can't
        // affect anything else that shares the process (the TUI / other tests) — see HarmonyPatches.
        GameHost host;
        HarmonyPatches.SkipMapPruning = true;
        try
        {
            host = GameHost.StartNewRun(seed, new[] { character }, ascension: 0);
            t = ResetProfiler.Mark("StartNewRun", t);
            host.EnterFirstRoom();
            t = ResetProfiler.Mark("EnterFirstRoom", t);
            ReachMap(host);
            t = ResetProfiler.Mark("ReachMap(Neow)", t);
        }
        finally
        {
            HarmonyPatches.SkipMapPruning = false;
        }

        Player player = host.Run.Players[0];
        NormalizeRelics(player, character);
        if (useStarterDeck)
        {
            SetStarterDeck(player, character);   // fixed starting deck; starter relic only, no randoms
        }
        else
        {
            List<CardModel> pool = ModelDb.AllCards
                .Where(c => c.VisualCardPool != null && c.VisualCardPool.Id.Entry == character.CardPool.Id.Entry)
                .ToList();
            NormalizeDeck(player, pool, rng);
            // Grant 5 distinct random relics that don't spawn a pickup reward (which would derail setup).
            var starterIds = character.StartingRelics.Select(r => r.Id.Entry).ToHashSet();
            foreach (RelicModel relic in ModelDb.AllRelics
                         .Where(r => !r.HasUponPickupEffect && !starterIds.Contains(r.Id.Entry))
                         .OrderBy(_ => rng.Next()).Take(RelicCount))
            {
                host.ObtainRelicDebug(relic);
            }
        }

        // Full HP last, so relic on-obtain max-HP changes don't leave us off a clean start.
        int startHp = character.StartingHp;
        player.Creature.SetMaxHpInternal(startHp);
        player.Creature.SetCurrentHpInternal(startHp);
        t = ResetProfiler.Mark("normalize(deck/relics/hp)", t);

        (EncounterModel encounter, int actIndex, RoomType roomType) = PickEncounter(rng, elitePct, bossPct, act);
        host.EnterEncounterDebug(encounter);
        ResetProfiler.Mark("EnterEncounterDebug", t);
        ResetProfiler.Done();

        var spec = new Spec(
            character.Id.Entry, encounter.GetType().Name, actIndex, roomType.ToString(),
            startHp, StarterHealFor(character.Id.Entry));
        return (host, spec);
    }

    /// <summary>
    /// <b>Soft reset</b>: reuse an already-set-up run (same character) and drop straight into a fresh
    /// randomized fight, skipping the ~160&#160;ms <see cref="GameHost.StartNewRun"/> +
    /// <see cref="GameHost.EnterFirstRoom"/> + opening-event teardown/rebuild. Re-normalizes the player
    /// (deck/relics/HP) and calls <see cref="GameHost.EnterEncounterDebug"/>, which cleanly tears down any
    /// prior combat and starts a new one — measured at ~1–2&#160;ms vs ~160&#160;ms for a full
    /// <see cref="Create"/>. The character is fixed by the run, so training spreads characters across env
    /// processes rather than re-rolling per fight.
    /// </summary>
    public static Spec Reenter(
        GameHost host, Random rng, double elitePct, double bossPct,
        bool useStarterDeck = false, int act = -1)
    {
        Player player = host.Run.Players[0];
        CharacterModel character = player.Character;

        // Tear down the previous fight's combat + map-history residue before re-entering, or it leaks
        // unboundedly (cards/piles/event subscriptions, and the score/map history) — see PrepareForSoftReenter.
        host.PrepareForSoftReenter();

        NormalizeRelics(player, character);
        if (useStarterDeck)
        {
            SetStarterDeck(player, character);
        }
        else
        {
            List<CardModel> pool = ModelDb.AllCards
                .Where(c => c.VisualCardPool != null && c.VisualCardPool.Id.Entry == character.CardPool.Id.Entry)
                .ToList();
            NormalizeDeck(player, pool, rng);
            var starterIds = character.StartingRelics.Select(r => r.Id.Entry).ToHashSet();
            foreach (RelicModel relic in ModelDb.AllRelics
                         .Where(r => !r.HasUponPickupEffect && !starterIds.Contains(r.Id.Entry))
                         .OrderBy(_ => rng.Next()).Take(RelicCount))
            {
                host.ObtainRelicDebug(relic);
            }
        }

        int startHp = character.StartingHp;
        player.Creature.SetMaxHpInternal(startHp);
        player.Creature.SetCurrentHpInternal(startHp);

        (EncounterModel encounter, int actIndex, RoomType roomType) = PickEncounter(rng, elitePct, bossPct, act);
        host.EnterEncounterDebug(encounter);

        return new Spec(
            character.Id.Entry, encounter.GetType().Name, actIndex, roomType.ToString(),
            startHp, StarterHealFor(character.Id.Entry));
    }

    private static void SetStarterDeck(Player player, CharacterModel character)
    {
        CardPile deck = player.Deck;
        foreach (CardModel c in deck.Cards.ToList())
        {
            deck.RemoveInternal(c, silent: true);
        }
        foreach (CardModel canon in character.StartingDeck)
        {
            CardModel card = canon.ToMutable();
            card.Owner = player;
            deck.AddInternal(card, deck.Cards.Count, silent: true);
        }
    }

    /// <summary>
    /// Create a fully-specified scenario for closed evals: an exact character, deck (card ids), optional
    /// extra relics (beyond the starter), and a named encounter — everything deterministic, so a fixed
    /// situation can be reproduced and the policy inspected on it.
    /// </summary>
    public static (GameHost Host, Spec Spec) CreateExplicit(
        string seed, string characterName, IReadOnlyList<string> cardIds,
        IReadOnlyList<string>? relicIds, string encounterName, IReadOnlyList<int>? enemyHp = null)
    {
        CharacterModel character = ModelDb.AllCharacters.First(
            c => c.Id.Entry.Contains(characterName, StringComparison.OrdinalIgnoreCase));

        GameHost host = GameHost.StartNewRun(seed, new[] { character }, ascension: 0);
        host.EnterFirstRoom();
        ReachMap(host);
        Player player = host.Run.Players[0];

        // Exact deck from the given ids.
        CardPile deck = player.Deck;
        foreach (CardModel c in deck.Cards.ToList())
        {
            deck.RemoveInternal(c, silent: true);
        }
        foreach (string id in cardIds)
        {
            CardModel card = ResolveCard(id).ToMutable();
            card.Owner = player;   // wires RunState too; the run's hook iteration NREs on an unowned card
            deck.AddInternal(card, deck.Cards.Count, silent: true);
        }

        NormalizeRelics(player, character);
        if (relicIds is not null)
        {
            foreach (string id in relicIds)
            {
                host.ObtainRelicDebug(ResolveRelic(id));
            }
        }

        int startHp = character.StartingHp;
        player.Creature.SetMaxHpInternal(startHp);
        player.Creature.SetCurrentHpInternal(startHp);

        (EncounterModel enc, int act) = ResolveEncounter(encounterName);
        host.EnterEncounterDebug(enc);

        // Optionally fix each enemy's HP so a scenario has an unambiguous best play (e.g. a free lethal).
        if (enemyHp is not null && host.Combat is { } combat)
        {
            var enemies = combat.Enemies.ToList();
            for (int i = 0; i < enemies.Count && i < enemyHp.Count; i++)
            {
                int hp = enemyHp[i];
                if (hp <= 0)
                {
                    continue;
                }
                if (hp > enemies[i].MaxHp)
                {
                    enemies[i].SetMaxHpInternal(hp);
                }
                enemies[i].SetCurrentHpInternal(hp);
            }
        }

        var spec = new Spec(character.Id.Entry, enc.GetType().Name, act, enc.RoomType.ToString(),
            startHp, StarterHealFor(character.Id.Entry));
        return (host, spec);
    }

    private static CardModel ResolveCard(string id) => ModelDb.AllCards.First(
        c => string.Equals(c.Id.Entry, id, StringComparison.OrdinalIgnoreCase)
             || string.Equals(c.GetType().Name, id, StringComparison.OrdinalIgnoreCase));

    private static RelicModel ResolveRelic(string id) => ModelDb.AllRelics.First(
        r => string.Equals(r.Id.Entry, id, StringComparison.OrdinalIgnoreCase)
             || string.Equals(r.GetType().Name, id, StringComparison.OrdinalIgnoreCase));

    private static (EncounterModel, int Act) ResolveEncounter(string name)
    {
        for (int act = 0; act < 3; act++)
        {
            EncounterModel? e = ModelDb.ActsByIndex[act].SelectMany(a => a.AllEncounters)
                .FirstOrDefault(e => string.Equals(e.GetType().Name, name, StringComparison.OrdinalIgnoreCase));
            if (e is not null)
            {
                return (e, act);
            }
        }
        throw new ArgumentException($"No encounter named '{name}'.");
    }

    private static CharacterModel PickCharacter(Random rng, string? name)
    {
        List<CharacterModel> all = ModelDb.AllCharacters.ToList();
        if (!string.IsNullOrEmpty(name))
        {
            return all.First(c => c.Id.Entry.Contains(name!, StringComparison.OrdinalIgnoreCase));
        }
        return all[rng.Next(all.Count)];
    }

    /// <summary>Take the opening ancient event's options until the run is on the map (a room-entry-ready
    /// state). Any blessing effects are overwritten by the normalization that follows.</summary>
    private static void ReachMap(GameHost host)
    {
        for (int guard = 0; guard < 12 && host.GetState().Phase != GamePhase.Map; guard++)
        {
            IReadOnlyList<GameOption> opts = host.ListOptions();
            GameOption? pick = opts.FirstOrDefault(o => o.Kind == OptionKind.ChooseEventOption)
                               ?? opts.FirstOrDefault();
            if (pick is null)
            {
                break;
            }
            host.Apply(pick);
        }
        if (host.GetState().Phase != GamePhase.Map)
        {
            throw new InvalidOperationException(
                $"Could not reach the map to set up a scenario (stuck on {host.GetState().Phase}).");
        }
    }

    private static void NormalizeDeck(Player player, List<CardModel> pool, Random rng)
    {
        CardPile deck = player.Deck;
        foreach (CardModel c in deck.Cards.ToList())
        {
            deck.RemoveInternal(c, silent: true);
        }
        for (int i = 0; i < DeckSize; i++)
        {
            CardModel card = pool[rng.Next(pool.Count)].ToMutable();
            card.Owner = player;   // required: the run's hook iteration NREs on a card with no owner
            deck.AddInternal(card, deck.Cards.Count, silent: true);
        }
    }

    private static void NormalizeRelics(Player player, CharacterModel character)
    {
        var starterIds = character.StartingRelics.Select(r => r.Id.Entry).ToHashSet();
        foreach (RelicModel r in player.Relics.Where(r => !starterIds.Contains(r.Id.Entry)).ToList())
        {
            player.RemoveRelicInternal(r, silent: true);
        }
    }

    private static (EncounterModel, int Act, RoomType) PickEncounter(
        Random rng, double elitePct, double bossPct, int actArg = -1)
    {
        int act = actArg >= 0 ? actArg : rng.Next(3); // fixed act, or 0/1/2
        double roll = rng.NextDouble();
        RoomType want = roll < bossPct ? RoomType.Boss
            : roll < bossPct + elitePct ? RoomType.Elite
            : RoomType.Monster;

        List<EncounterModel> encs = ModelDb.ActsByIndex[act]
            .SelectMany(a => a.AllEncounters)
            .Where(e => e.RoomType == want)
            .ToList();
        if (encs.Count == 0)
        {
            // Fall back to a normal fight if the act has none of the wanted type.
            want = RoomType.Monster;
            encs = ModelDb.ActsByIndex[act].SelectMany(a => a.AllEncounters)
                .Where(e => e.RoomType == want).ToList();
        }
        return (encs[rng.Next(encs.Count)], act, want);
    }
}

/// <summary>
/// Opt-in per-phase timing of a full scenario <see cref="CombatScenario.Create"/> (set
/// <c>LTS2_PROFILE=1</c>): accumulates how long each phase of a full reset takes — StartNewRun,
/// EnterFirstRoom, ReachMap (the opening Neow event), deck/relic/HP normalize, and EnterEncounterDebug —
/// and prints an average ms breakdown to stderr every N resets. Full reset (~160&#160;ms) dominates
/// scenario iteration time; this shows which phase to optimize. Zero cost when off.
/// </summary>
internal static class ResetProfiler
{
    private static readonly bool Enabled =
        Environment.GetEnvironmentVariable("LTS2_PROFILE") is "1" or "true";
    private const int ReportEvery = 25;

    private static readonly object Lock = new();
    private static readonly Dictionary<string, long> Ticks = new();
    private static readonly List<string> Order = new();
    private static int _resets;

    public static long Mark(string phase, long since)
    {
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (Enabled)
        {
            lock (Lock)
            {
                if (!Ticks.ContainsKey(phase))
                {
                    Order.Add(phase);
                }
                Ticks.TryGetValue(phase, out long acc);
                Ticks[phase] = acc + (now - since);
            }
        }
        return now;
    }

    public static void Done()
    {
        if (!Enabled)
        {
            return;
        }
        lock (Lock)
        {
            if (++_resets % ReportEvery != 0)
            {
                return;
            }
            double msPerReset = 1000.0 / System.Diagnostics.Stopwatch.Frequency / _resets;
            long total = Ticks.Values.Sum();
            string parts = string.Join("  ", Order.Select(p => $"{p}={Ticks[p] * msPerReset:0.0}ms"));
            Console.Error.WriteLine(
                $"[reset-profile] resets={_resets} total={total * msPerReset:0.0}ms/reset :: {parts}");
        }
    }
}
