using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Lts2.Agent;
using Lts2.Agent.Wire;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using Xunit;

namespace Lts2.Harness.Tests;

/// <summary>
/// Tests for the <c>reset_combat</c> <c>deckSpec</c> field (roadmap M1 / contract 3): the realistic
/// deck sampler's determinism (same seed + same spec ⇒ byte-identical deck), its size bounds and
/// no-status-cards guarantee, and that the explicit and absent-spec paths still behave as before. All
/// drive the real <see cref="TrainingEnvironmentServer"/> over an in-memory channel with fixed seeds.
/// </summary>
public sealed class DeckSpecTests
{
    /// <summary>Run one reset_combat command through the server and return its parsed observation.</summary>
    private static JsonDocument ResetCombat(string commandJson)
    {
        string input = commandJson + "\n" + """{"cmd":"close"}""" + "\n";
        var output = new System.IO.StringWriter();
        var channel = new StreamLineChannel(new System.IO.StringReader(input), output);
        new TrainingEnvironmentServer().Serve(channel);

        string[] lines = output.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.True(lines.Length >= 1, "expected at least the reset_combat observation");
        JsonDocument doc = JsonDocument.Parse(lines[0]);
        Assert.False(doc.RootElement.TryGetProperty("error", out _), lines[0]);
        return doc;
    }

    /// <summary>Run one reset_combat command per seed through a <b>single</b> server session (so the game
    /// runtime boots once) and return each parsed observation. Used to sweep the relic/potion sampling over
    /// many seeds without paying the boot cost per call.</summary>
    private static List<JsonDocument> ResetCombatMany(IEnumerable<string> seeds, string specJson)
    {
        var sb = new System.Text.StringBuilder();
        int count = 0;
        foreach (string seed in seeds)
        {
            sb.Append("""{"cmd":"reset_combat","seed":""")
              .Append(JsonSerializer.Serialize(seed))
              .Append(""","character":"Iron","act":0,"deckSpec":""")
              .Append(specJson)
              .Append("}\n");
            count++;
        }
        sb.Append("""{"cmd":"close"}""").Append('\n');

        var output = new System.IO.StringWriter();
        var channel = new StreamLineChannel(new System.IO.StringReader(sb.ToString()), output);
        new TrainingEnvironmentServer().Serve(channel);

        string[] lines = output.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var docs = new List<JsonDocument>(count);
        for (int i = 0; i < count; i++)
        {
            JsonDocument doc = JsonDocument.Parse(lines[i]);
            Assert.False(doc.RootElement.TryGetProperty("error", out _), lines[i]);
            docs.Add(doc);
        }
        return docs;
    }

    private static List<string> InfoStringList(JsonElement root, string prop)
    {
        JsonElement info = root.GetProperty("info");
        Assert.True(info.TryGetProperty(prop, out JsonElement arr), $"info.{prop} missing");
        return arr.EnumerateArray().Select(e => e.GetString()!).ToList();
    }

    /// <summary>The relic ids the player currently holds (state projection).</summary>
    private static List<string> PlayerRelicIds(JsonElement root) =>
        root.GetProperty("state").GetProperty("players")[0].GetProperty("relics")
            .EnumerateArray().Select(e => e.GetString()!).ToList();

    /// <summary>The potion ids the player currently holds (non-empty belt slots only).</summary>
    private static List<string> PlayerPotionIds(JsonElement root) =>
        root.GetProperty("state").GetProperty("players")[0].GetProperty("potions")
            .EnumerateArray().Where(e => e.ValueKind != JsonValueKind.Null).Select(e => e.GetString()!).ToList();

    /// <summary>The card ids across the four combat piles, in pile+position order — the built deck plus
    /// its (deterministic) shuffle. At the opening observation this is exactly the deck we built.</summary>
    private static List<string> DeckCardIds(JsonElement root)
    {
        JsonElement combat = root.GetProperty("state").GetProperty("players")[0].GetProperty("combatState");
        var ids = new List<string>();
        foreach (string pile in new[] { "hand", "drawPile", "discardPile", "exhaustPile" })
        {
            foreach (JsonElement card in combat.GetProperty(pile).EnumerateArray())
            {
                ids.Add(card.GetProperty("cardId").GetString()!);
            }
        }
        return ids;
    }

    private static List<string> DeckCardTypes(JsonElement root)
    {
        JsonElement combat = root.GetProperty("state").GetProperty("players")[0].GetProperty("combatState");
        var types = new List<string>();
        foreach (string pile in new[] { "hand", "drawPile", "discardPile", "exhaustPile" })
        {
            foreach (JsonElement card in combat.GetProperty(pile).EnumerateArray())
            {
                types.Add(card.GetProperty("type").GetString()!);
            }
        }
        return types;
    }

    [Fact]
    public void Realistic_SameSeedSameSpec_ProducesIdenticalDeck()
    {
        const string cmd =
            """{"cmd":"reset_combat","seed":"DECKSEED","character":"Iron","act":0,"deckSpec":{"kind":"realistic"}}""";

        using JsonDocument a = ResetCombat(cmd);
        using JsonDocument b = ResetCombat(cmd);

        List<string> deckA = DeckCardIds(a.RootElement);
        List<string> deckB = DeckCardIds(b.RootElement);

        Assert.NotEmpty(deckA);
        Assert.Equal(deckA, deckB);   // identical ids AND identical order (same shuffle)

        // The resolved sampler picks are surfaced and stable too — cards, relics, potions, starter state.
        JsonElement infoA = a.RootElement.GetProperty("info");
        JsonElement infoB = b.RootElement.GetProperty("info");
        Assert.Equal("realistic", infoA.GetProperty("deckSpec").GetString());
        Assert.Equal(InfoStringList(a.RootElement, "addedCards"), InfoStringList(b.RootElement, "addedCards"));
        Assert.Equal(InfoStringList(a.RootElement, "removedCards"), InfoStringList(b.RootElement, "removedCards"));
        Assert.Equal(InfoStringList(a.RootElement, "addedRelics"), InfoStringList(b.RootElement, "addedRelics"));
        Assert.Equal(InfoStringList(a.RootElement, "addedPotions"), InfoStringList(b.RootElement, "addedPotions"));
        Assert.Equal(
            infoA.GetProperty("starterRelicState").GetString(),
            infoB.GetProperty("starterRelicState").GetString());
        // And the actual granted relics/potions on the player match.
        Assert.Equal(PlayerRelicIds(a.RootElement), PlayerRelicIds(b.RootElement));
        Assert.Equal(PlayerPotionIds(a.RootElement), PlayerPotionIds(b.RootElement));
    }

    [Fact]
    public void Realistic_RespectsSizeBounds_AndDealsNoStatusCards()
    {
        GameRuntime.EnsureInitialized();
        CharacterModel ironclad = ModelDb.AllCharacters.First(
            c => c.Id.Entry.Contains("IRONCLAD", StringComparison.OrdinalIgnoreCase));
        int starterCount = ironclad.StartingDeck.Count();

        // Force the maximal add/remove so the bounds arithmetic is exercised at its widest. Pin relics and
        // potions to [0,0] so the deck-size / no-status assertions isolate the DECK — a granted relic can
        // inject combat-start cards (incl. status) into the opening piles, which is realistic but would break
        // the exact "starter − removals + additions" count this test verifies.
        const string cmd =
            """{"cmd":"reset_combat","seed":"BOUNDS1","character":"Iron","act":0,"deckSpec":{"kind":"realistic","removals":[0,3],"additions":[3,3],"relics":[0,0],"potions":[0,0],"starterRelic":{"absent":0,"orobas":0}}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;
        JsonElement info = root.GetProperty("info");

        var removed = info.GetProperty("removedCards").EnumerateArray().Select(e => e.GetString()!).ToList();
        var added = info.GetProperty("addedCards").EnumerateArray().Select(e => e.GetString()!).ToList();

        Assert.InRange(removed.Count, 0, 3);
        Assert.Equal(3, added.Count);   // additions:[3,3] is exactly 3

        // Deck size = starter − removals + additions.
        List<string> deck = DeckCardIds(root);
        Assert.Equal(starterCount - removed.Count + added.Count, deck.Count);

        // No status-type card was dealt into the deck.
        Assert.DoesNotContain(CardType.Status.ToString(), DeckCardTypes(root));

        // And no sampled addition resolves to a status card (the sampler's own guarantee).
        foreach (string id in added)
        {
            CardModel card = ModelDb.AllCards.First(c => c.Id.Entry == id);
            Assert.NotEqual(CardType.Status, card.Type);
        }
    }

    [Fact]
    public void Realistic_ZeroRanges_ReproducesStarterRelicOnlyBehavior()
    {
        GameRuntime.EnsureInitialized();
        CharacterModel ironclad = ModelDb.AllCharacters.First(
            c => c.Id.Entry.Contains("IRONCLAD", StringComparison.OrdinalIgnoreCase));
        int starterCount = ironclad.StartingDeck.Count();
        int starterRelicCount = ironclad.StartingRelics.Count();

        // All four ranges zero: the previous v1 realistic behavior — starter deck, starter relic only, no
        // random relics or potions, and (because a [k,k] range consumes no rng) an unperturbed stream.
        const string cmd =
            """{"cmd":"reset_combat","seed":"ZERO1","character":"Iron","act":0,"deckSpec":{"kind":"realistic","removals":[0,0],"additions":[0,0],"relics":[0,0],"potions":[0,0],"starterRelic":{"absent":0,"orobas":0}}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;
        JsonElement info = root.GetProperty("info");

        Assert.Empty(info.GetProperty("removedCards").EnumerateArray());
        Assert.Empty(info.GetProperty("addedCards").EnumerateArray());
        Assert.Empty(info.GetProperty("addedRelics").EnumerateArray());
        Assert.Empty(info.GetProperty("addedPotions").EnumerateArray());
        Assert.Equal(starterCount, DeckCardIds(root).Count);
        // Only the starter relic(s), and an empty potion belt.
        Assert.Equal(starterRelicCount, PlayerRelicIds(root).Count);
        Assert.Empty(PlayerPotionIds(root));
    }

    [Fact]
    public void ExplicitDeckSpec_BuildsTheGivenDeck()
    {
        // The explicit deckSpec form mirrors the legacy top-level `cards` closed-eval path.
        const string cmd =
            """{"cmd":"reset_combat","seed":"EXP1","character":"Ironclad","encounter":"CultistsNormal","deckSpec":{"kind":"explicit","cards":["STRIKE_IRONCLAD","STRIKE_IRONCLAD","DEFEND_IRONCLAD"]}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;

        Assert.Equal("explicit", root.GetProperty("info").GetProperty("deckSpec").GetString());
        List<string> deck = DeckCardIds(root);
        Assert.Equal(3, deck.Count);
        Assert.Equal(2, deck.Count(id => id == "STRIKE_IRONCLAD"));
        Assert.Equal(1, deck.Count(id => id == "DEFEND_IRONCLAD"));
    }

    [Fact]
    public void AbsentDeckSpec_MatchesLegacyRandomFifteen()
    {
        // No deckSpec: exactly today's behavior — a random 15-card deck, tagged "random".
        const string cmd = """{"cmd":"reset_combat","seed":"LEGACY1","character":"Iron","act":0}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;

        Assert.Equal("random", root.GetProperty("info").GetProperty("deckSpec").GetString());
        Assert.Equal(15, DeckCardIds(root).Count);
        // removedCards/addedCards are omitted (null) for non-realistic decks.
        Assert.False(root.GetProperty("info").TryGetProperty("removedCards", out _));
    }

    [Fact]
    public void RandomDeckSpec_HonorsExplicitCardCount()
    {
        const string cmd =
            """{"cmd":"reset_combat","seed":"RND1","character":"Iron","act":0,"deckSpec":{"kind":"random","cards":10}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;

        Assert.Equal("random", root.GetProperty("info").GetProperty("deckSpec").GetString());
        Assert.Equal(10, DeckCardIds(root).Count);
    }

    [Fact]
    public void Realistic_RelicAndPotionCounts_WithinDefaultRanges_AllValuesObserved()
    {
        // Default realistic ranges: relics [0,2], potions [0,1]. Sweep fixed seeds and confirm every count
        // lands in range and all values are observed. Pin the starter relic to normal so the random-relic
        // count is unambiguous (Orobas would add the upgrade + Touch of Orobas on top).
        var seeds = Enumerable.Range(0, 32).Select(i => $"COUNTS{i}");
        List<JsonDocument> docs = ResetCombatMany(
            seeds, """{"kind":"realistic","starterRelic":{"absent":0,"orobas":0}}""");
        try
        {
            var relicCounts = new HashSet<int>();
            var potionCounts = new HashSet<int>();
            foreach (JsonDocument d in docs)
            {
                int relics = InfoStringList(d.RootElement, "addedRelics").Count;
                int potions = InfoStringList(d.RootElement, "addedPotions").Count;
                Assert.InRange(relics, 0, 2);
                Assert.InRange(potions, 0, 1);
                relicCounts.Add(relics);
                potionCounts.Add(potions);
            }
            Assert.Equal(new HashSet<int> { 0, 1, 2 }, relicCounts);
            Assert.Equal(new HashSet<int> { 0, 1 }, potionCounts);
        }
        finally
        {
            docs.ForEach(d => d.Dispose());
        }
    }

    [Fact]
    public void Realistic_NeverGrantsHpPotion_AndExclusionFilterIsNonEmpty()
    {
        GameRuntime.EnsureInitialized();

        // The HP-exclusion filter must actually exclude real potions (the game has such potions).
        IReadOnlyList<string> excluded = PotionCatalog.HpExcludedPotionIds();
        Assert.NotEmpty(excluded);
        foreach (string id in new[] { "BLOOD_POTION", "FRUIT_JUICE", "REGEN_POTION", "FAIRY_IN_A_BOTTLE" })
        {
            Assert.Contains(id, excluded);
        }
        var excludedSet = excluded.ToHashSet();

        // Force exactly one potion per fight over many seeds; none may ever be an HP potion.
        var seeds = Enumerable.Range(0, 32).Select(i => $"HPPOT{i}");
        List<JsonDocument> docs = ResetCombatMany(
            seeds, """{"kind":"realistic","relics":[0,0],"potions":[1,1],"starterRelic":{"absent":0,"orobas":0}}""");
        try
        {
            int granted = 0;
            foreach (JsonDocument d in docs)
            {
                List<string> potions = InfoStringList(d.RootElement, "addedPotions");
                Assert.Single(potions);           // potions:[1,1] grants exactly one
                Assert.DoesNotContain(potions[0], excludedSet);
                granted++;
            }
            Assert.True(granted > 0, "expected potions to be granted");
        }
        finally
        {
            docs.ForEach(d => d.Dispose());
        }
    }

    [Fact]
    public void Realistic_StarterRelicStates_AllObservedOverManySeeds()
    {
        // Elevated probabilities so all three states appear over a modest sweep (the default 10/10 is
        // exercised by the determinism test); the sampling logic is identical.
        var seeds = Enumerable.Range(0, 40).Select(i => $"STATE{i}");
        List<JsonDocument> docs = ResetCombatMany(
            seeds, """{"kind":"realistic","relics":[0,0],"potions":[0,0],"starterRelic":{"absent":0.34,"orobas":0.33}}""");
        try
        {
            var states = docs
                .Select(d => d.RootElement.GetProperty("info").GetProperty("starterRelicState").GetString())
                .ToHashSet();
            Assert.Equal(new HashSet<string?> { "normal", "absent", "orobas" }, states);
        }
        finally
        {
            docs.ForEach(d => d.Dispose());
        }
    }

    [Fact]
    public void Realistic_Orobas_AddsUpgradedAndTouchOfOrobas_RemovesBaseStarter()
    {
        // Force the Orobas state: the Ironclad's Burning Blood is replaced by Black Blood AND Touch of Orobas
        // is granted (the game's own ancient-reward effect).
        const string cmd =
            """{"cmd":"reset_combat","seed":"OROB1","character":"Iron","act":0,"deckSpec":{"kind":"realistic","relics":[0,0],"potions":[0,0],"starterRelic":{"absent":0,"orobas":1}}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;
        JsonElement info = root.GetProperty("info");

        Assert.Equal("orobas", info.GetProperty("starterRelicState").GetString());
        string upgraded = info.GetProperty("upgradedStarterRelic").GetString()!;
        Assert.Equal("BLACK_BLOOD", upgraded);

        List<string> relics = PlayerRelicIds(root);
        Assert.Contains(upgraded, relics);
        Assert.Contains("TOUCH_OF_OROBAS", relics);
        Assert.DoesNotContain("BURNING_BLOOD", relics);   // base starter relic was replaced
    }

    [Fact]
    public void Realistic_Absent_GrantsNoStarterRelic()
    {
        const string cmd =
            """{"cmd":"reset_combat","seed":"ABS1","character":"Iron","act":0,"deckSpec":{"kind":"realistic","relics":[0,0],"potions":[0,0],"starterRelic":{"absent":1,"orobas":0}}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement root = doc.RootElement;

        Assert.Equal("absent", root.GetProperty("info").GetProperty("starterRelicState").GetString());
        List<string> relics = PlayerRelicIds(root);
        Assert.DoesNotContain("BURNING_BLOOD", relics);
        Assert.Empty(relics);   // relics:[0,0] and no starter relic => no relics at all
    }

    /// <summary>The relic ids the player currently holds, read straight off the live run.</summary>
    private static List<string> LiveRelicIds(GameHost host) =>
        host.Run.Players[0].Relics.Select(r => r.Id.Entry).ToList();

    private static void AssertNoDuplicateRelics(GameHost host, int fight)
    {
        List<string> relics = LiveRelicIds(host);
        List<string> dups = relics.GroupBy(id => id).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
        Assert.True(dups.Count == 0, $"fight {fight}: duplicate relic ids [{string.Join(", ", dups)}] in [{string.Join(", ", relics)}]");
    }

    [Fact]
    public void Reenter_Orobas_NoDuplicateRelics_AndExpectedComposition_AcrossManyReenters()
    {
        GameRuntime.EnsureInitialized();
        CombatScenario.PoolWeights w = CombatScenario.DeckSpec.DefaultWeights;
        // Force Orobas every fight, plus one random relic. Expected per fight: upgraded starter (BLACK_BLOOD)
        // + TOUCH_OF_OROBAS + the granted random — no plain starter, and no duplicate ids.
        var spec = new CombatScenario.DeckSpec.Realistic(0, 0, 0, 0, w, 1, 1, 0, 0, 0.0, 1.0);
        var rng = new Random(4242);

        (GameHost host, CombatScenario.Spec first) = CombatScenario.Create(
            "REORO", rng, "IRONCLAD", elitePct: 0.0, bossPct: 0.0, act: 0, deckSpec: spec);
        AssertOrobasComposition(host, first, 0);

        for (int i = 1; i <= 6; i++)
        {
            CombatScenario.Spec s = CombatScenario.Reenter(host, rng, 0.0, 0.0, act: 0, deckSpec: spec);
            AssertOrobasComposition(host, s, i);
        }
    }

    private static void AssertOrobasComposition(GameHost host, CombatScenario.Spec s, int fight)
    {
        AssertNoDuplicateRelics(host, fight);
        Assert.Equal("orobas", s.StarterRelicState);
        List<string> relics = LiveRelicIds(host);
        var expected = new HashSet<string>(s.AddedRelics!) { "TOUCH_OF_OROBAS", s.UpgradedStarterRelicId! };
        Assert.True(expected.SetEquals(relics),
            $"fight {fight}: relics [{string.Join(", ", relics)}] != expected [{string.Join(", ", expected)}]");
        Assert.DoesNotContain("BURNING_BLOOD", relics);   // plain starter replaced, never lingers
    }

    [Fact]
    public void Reenter_BroadRandom_NoDuplicateRelics_AndStableCount_AcrossManyReenters()
    {
        GameRuntime.EnsureInitialized();
        CharacterModel ironclad = ModelDb.AllCharacters.First(
            c => c.Id.Entry.Contains("IRONCLAD", StringComparison.OrdinalIgnoreCase));
        int starterRelicCount = ironclad.StartingRelics.Count();

        // Broad-random path: starter relic(s) + 5 random relics, every fight.
        var spec = new CombatScenario.DeckSpec.Random(15);
        var rng = new Random(1717);

        (GameHost host, CombatScenario.Spec _) = CombatScenario.Create(
            "RERND", rng, "IRONCLAD", elitePct: 0.0, bossPct: 0.0, act: 0, deckSpec: spec);
        AssertNoDuplicateRelics(host, 0);
        Assert.Equal(starterRelicCount + 5, LiveRelicIds(host).Count);

        for (int i = 1; i <= 6; i++)
        {
            CombatScenario.Reenter(host, rng, 0.0, 0.0, act: 0, deckSpec: spec);
            AssertNoDuplicateRelics(host, i);
            Assert.Equal(starterRelicCount + 5, LiveRelicIds(host).Count);
        }
    }

    [Fact]
    public void Realistic_StarterHeal_FollowsStarterRelicState()
    {
        // StarterHeal feeds the hpLost accounting (the starter relic's end-of-combat heal is added back on a
        // win). Drive CombatScenario directly so the internal Spec.StarterHeal is observable per state.
        // Ironclad: normal = Burning Blood (6), orobas = Black Blood (12), absent = 0.
        GameRuntime.EnsureInitialized();
        CombatScenario.PoolWeights w = CombatScenario.DeckSpec.DefaultWeights;

        (string State, int Heal, string? Upgraded) Run(double absent, double orobas)
        {
            var spec = new CombatScenario.DeckSpec.Realistic(0, 0, 0, 0, w, 0, 0, 0, 0, absent, orobas);
            (GameHost _, CombatScenario.Spec s) = CombatScenario.Create(
                "HEALSEED", new Random(7), "IRONCLAD", elitePct: 0.0, bossPct: 0.0, act: 0, deckSpec: spec);
            return (s.StarterRelicState!, s.StarterHeal, s.UpgradedStarterRelicId);
        }

        (string State, int Heal, string? Upgraded) normal = Run(0.0, 0.0);
        Assert.Equal("normal", normal.State);
        Assert.Equal(6, normal.Heal);

        (string State, int Heal, string? Upgraded) absent = Run(1.0, 0.0);
        Assert.Equal("absent", absent.State);
        Assert.Equal(0, absent.Heal);

        (string State, int Heal, string? Upgraded) orobas = Run(0.0, 1.0);
        Assert.Equal("orobas", orobas.State);
        Assert.Equal(12, orobas.Heal);
        Assert.Equal("BLACK_BLOOD", orobas.Upgraded);
    }
}
