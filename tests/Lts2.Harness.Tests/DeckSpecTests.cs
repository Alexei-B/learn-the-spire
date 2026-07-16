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

        // The resolved sampler picks are surfaced and stable too.
        JsonElement infoA = a.RootElement.GetProperty("info");
        JsonElement infoB = b.RootElement.GetProperty("info");
        Assert.Equal("realistic", infoA.GetProperty("deckSpec").GetString());
        Assert.Equal(
            infoA.GetProperty("addedCards").EnumerateArray().Select(e => e.GetString()).ToList(),
            infoB.GetProperty("addedCards").EnumerateArray().Select(e => e.GetString()).ToList());
        Assert.Equal(
            infoA.GetProperty("removedCards").EnumerateArray().Select(e => e.GetString()).ToList(),
            infoB.GetProperty("removedCards").EnumerateArray().Select(e => e.GetString()).ToList());
    }

    [Fact]
    public void Realistic_RespectsSizeBounds_AndDealsNoStatusCards()
    {
        GameRuntime.EnsureInitialized();
        CharacterModel ironclad = ModelDb.AllCharacters.First(
            c => c.Id.Entry.Contains("IRONCLAD", StringComparison.OrdinalIgnoreCase));
        int starterCount = ironclad.StartingDeck.Count();

        // Force the maximal add/remove so the bounds arithmetic is exercised at its widest.
        const string cmd =
            """{"cmd":"reset_combat","seed":"BOUNDS1","character":"Iron","act":0,"deckSpec":{"kind":"realistic","removals":[0,3],"additions":[3,3]}}""";
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
    public void Realistic_ZeroRangesDegradeToStarterDeck()
    {
        GameRuntime.EnsureInitialized();
        CharacterModel ironclad = ModelDb.AllCharacters.First(
            c => c.Id.Entry.Contains("IRONCLAD", StringComparison.OrdinalIgnoreCase));
        int starterCount = ironclad.StartingDeck.Count();

        const string cmd =
            """{"cmd":"reset_combat","seed":"ZERO1","character":"Iron","act":0,"deckSpec":{"kind":"realistic","removals":[0,0],"additions":[0,0]}}""";
        using JsonDocument doc = ResetCombat(cmd);
        JsonElement info = doc.RootElement.GetProperty("info");

        Assert.Empty(info.GetProperty("removedCards").EnumerateArray());
        Assert.Empty(info.GetProperty("addedCards").EnumerateArray());
        Assert.Equal(starterCount, DeckCardIds(doc.RootElement).Count);
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
}
