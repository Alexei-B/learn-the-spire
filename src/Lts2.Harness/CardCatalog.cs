using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.CardPools;

namespace Lts2.Harness;

/// <summary>
/// Single source of truth for classifying a card by the pool it belongs to (character / colorless /
/// curse / status / …) and for enumerating the sampling pools the realistic deck generator draws from.
/// Both the <see cref="CombatScenario"/> realistic sampler and the <c>--dump-cards</c> catalog dump read
/// their pool metadata from here, so the flags Python sees match the pools training actually samples.
/// </summary>
public static class CardCatalog
{
    /// <summary>Which broad pool a card lives in. Character = one of the five playable characters' own
    /// pools; the rest are the shared pools. Status cards are never dealt into a training deck.</summary>
    public enum PoolCategory
    {
        Character,
        Colorless,
        Curse,
        Status,
        Event,
        Quest,
        Token,
        Other,
    }

    /// <summary>Classify a card by its (visual-agnostic) home pool. Guards the game's <see cref="CardModel.Pool"/>
    /// lookup, which throws for the rare card that is in no pool — those fall back to a type-based guess.</summary>
    public static PoolCategory CategoryOf(CardModel card)
    {
        if (card is null)
        {
            throw new ArgumentNullException(nameof(card));
        }

        CardPoolModel? pool = null;
        try
        {
            pool = card.Pool;
        }
        catch
        {
            // Some models are not registered in any pool; fall through to the type-based classification.
        }

        return pool switch
        {
            ColorlessCardPool => PoolCategory.Colorless,
            CurseCardPool => PoolCategory.Curse,
            StatusCardPool => PoolCategory.Status,
            EventCardPool => PoolCategory.Event,
            QuestCardPool => PoolCategory.Quest,
            TokenCardPool => PoolCategory.Token,
            not null when IsCharacterPool(pool) => PoolCategory.Character,
            _ => card.Type switch
            {
                CardType.Status => PoolCategory.Status,
                CardType.Curse => PoolCategory.Curse,
                _ => PoolCategory.Other,
            },
        };
    }

    /// <summary>The card's home pool title (e.g. <c>"ironclad"</c>, <c>"colorless"</c>, <c>"curse"</c>,
    /// <c>"status"</c>), or null if the card is in no pool.</summary>
    public static string? PoolTitle(CardModel card)
    {
        try
        {
            return card.Pool.Title;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>True if a card should never be dealt into any deck by any deck spec (status cards).</summary>
    public static bool IsDealable(CardModel card) =>
        card.Type != CardType.Status && CategoryOf(card) != PoolCategory.Status;

    /// <summary>The character's own card pool, minus anything not dealable (status).</summary>
    public static IReadOnlyList<CardModel> OwnPool(CharacterModel character) =>
        character.CardPool.AllCards.Where(IsDealable).ToList();

    /// <summary>The shared colorless pool, dealable cards only.</summary>
    public static IReadOnlyList<CardModel> ColorlessPool() =>
        ModelDb.CardPool<ColorlessCardPool>().AllCards.Where(IsDealable).ToList();

    /// <summary>The shared curse pool.</summary>
    public static IReadOnlyList<CardModel> CursePool() =>
        ModelDb.CardPool<CurseCardPool>().AllCards.Where(IsDealable).ToList();

    /// <summary>Every other playable character's own pool (dealable cards only) — the "off-character"
    /// sampling pool.</summary>
    public static IReadOnlyList<CardModel> OffCharacterPool(CharacterModel character)
    {
        string ownPoolId = character.CardPool.Id.Entry;
        return ModelDb.AllCharacters
            .Where(c => c.CardPool.Id.Entry != ownPoolId)
            .SelectMany(c => c.CardPool.AllCards)
            .Where(IsDealable)
            .ToList();
    }

    private static bool IsCharacterPool(CardPoolModel pool)
    {
        string id = pool.Id.Entry;
        return ModelDb.AllCharacters.Any(c => c.CardPool.Id.Entry == id);
    }
}
