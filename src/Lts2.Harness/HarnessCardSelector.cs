using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.TestSupport;

namespace Lts2.Harness;

/// <summary>
/// A card choice the game has requested mid-effect and is now blocked waiting on.
/// Surfaced to the harness so it can be resolved through <see cref="GameHost.ListOptions(ulong)"/>
/// / <see cref="GameHost.Apply"/> instead of a UI screen.
/// </summary>
public sealed class PendingChoice
{
    /// <summary>The cards the player may choose from, in stable order.</summary>
    public IReadOnlyList<CardModel> Options { get; }

    /// <summary>Minimum number of cards that must be selected (0 means the choice can be skipped).</summary>
    public int MinSelect { get; }

    /// <summary>Maximum number of cards that may be selected.</summary>
    public int MaxSelect { get; }

    /// <summary>
    /// True when this choice picks a card to *upgrade* (the rest-site forge), so the options can be
    /// shown as the upgraded card they would become rather than their current form.
    /// </summary>
    public bool IsUpgradeSelection { get; internal set; }

    internal TaskCompletionSource<IReadOnlyList<CardModel>> Completion { get; } =
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    internal PendingChoice(IReadOnlyList<CardModel> options, int minSelect, int maxSelect)
    {
        Options = options;
        MinSelect = minSelect;
        MaxSelect = maxSelect;
    }
}

/// <summary>
/// The harness's implementation of the game's <see cref="ICardSelector"/> seam
/// (<c>CardSelectCmd.Selector</c>). When the game requests a card selection mid-effect it
/// calls <see cref="GetSelectedCards"/>, which records a <see cref="PendingChoice"/>, signals
/// the harness, and blocks the effect's task until the harness resolves it. This is what lets
/// discover/scry/exhaust-style decisions surface through the public option API rather than
/// blocking on a UI screen.
///
/// The game runs each effect on a thread-pool continuation, so blocking here does not block
/// the harness thread: the harness waits on <see cref="PendingSignal"/> (or queue drain),
/// then calls <see cref="Resolve"/>, which completes the effect's task on the thread pool.
/// </summary>
internal sealed class HarnessCardSelector : ICardSelector
{
    private readonly object _gate = new();
    private PendingChoice? _pending;
    private TaskCompletionSource _pendingSignal = NewSignal();

    /// <summary>The choice the game is currently blocked on, or null if none.</summary>
    public PendingChoice? Pending
    {
        get { lock (_gate) { return _pending; } }
    }

    /// <summary>
    /// A task that completes when a choice becomes pending. Re-read after each
    /// <see cref="Resolve"/>: it is replaced with a fresh, uncompleted task each time a choice
    /// is resolved so the harness can wait for the next one.
    /// </summary>
    public Task PendingSignal
    {
        get { lock (_gate) { return _pendingSignal.Task; } }
    }

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var list = options.ToList();
        var choice = new PendingChoice(list, minSelect, maxSelect);
        lock (_gate)
        {
            if (_pending is not null)
            {
                throw new InvalidOperationException(
                    "A card choice is already pending; the harness resolves choices one at a time.");
            }
            _pending = choice;
            _pendingSignal.TrySetResult();
        }
        return AwaitSelection(choice);
    }

    private static async Task<IEnumerable<CardModel>> AwaitSelection(PendingChoice choice) =>
        await choice.Completion.Task;

    /// <summary>
    /// Resolve the pending choice with the given cards (which must be drawn from the choice's
    /// options). Resets the pending signal first so a follow-on choice on the resumed effect is
    /// observed cleanly, then unblocks the effect.
    /// </summary>
    public void Resolve(IReadOnlyList<CardModel> selected)
    {
        PendingChoice choice;
        lock (_gate)
        {
            choice = _pending ?? throw new InvalidOperationException("No choice is pending to resolve.");
            _pending = null;
            _pendingSignal = NewSignal();
        }
        choice.Completion.TrySetResult(selected);
    }

    /// <summary>
    /// The card the harness has chosen for the next post-combat card reward. Set by
    /// <see cref="GameHost.Apply"/> immediately before taking a <see cref="RewardType.Card"/> reward,
    /// then consumed by <see cref="GetSelectedCardReward"/>. Cleared after each read.
    /// </summary>
    internal CardModel? NextCardRewardPick { get; set; }

    /// <summary>
    /// The id (<see cref="CardRewardAlternative.OptionId"/>) of the card-reward *alternative* the
    /// harness has chosen instead of a card (e.g. "SACRIFICE"). Set by <see cref="GameHost.Apply"/>
    /// before running a terminal alternative through the rewards synchronizer, then consumed by
    /// <see cref="GetSelectedCardReward"/>. Matched by id (not reference) because the game regenerates
    /// the alternative list each selection round and matches the result by reference.
    /// </summary>
    internal string? NextCardRewardAlternativeId { get; set; }

    /// <summary>
    /// Card-reward selection (post-combat). Returns the staged alternative
    /// (<see cref="NextCardRewardAlternativeId"/>) when set — resolved against the live
    /// <paramref name="alternatives"/> by id — otherwise whichever card the harness pre-selected via
    /// <see cref="NextCardRewardPick"/>. Falls back to the first option if nothing was staged, so the
    /// flow never blocks on a missing choice.
    /// </summary>
    public CardRewardSelection GetSelectedCardReward(
        IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        if (NextCardRewardAlternativeId is { } altId)
        {
            NextCardRewardAlternativeId = null;
            CardRewardAlternative? alt = alternatives.FirstOrDefault(a => a.OptionId == altId);
            if (alt is not null)
            {
                NextCardRewardPick = null;
                return new CardRewardSelection { alternative = alt };
            }
        }
        CardModel? pick = NextCardRewardPick ?? options.FirstOrDefault()?.Card;
        NextCardRewardPick = null;
        return new CardRewardSelection { card = pick };
    }

    private static TaskCompletionSource NewSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);
}
