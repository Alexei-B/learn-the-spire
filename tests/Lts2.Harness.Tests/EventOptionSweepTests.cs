using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;
using Lts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Lts2.Harness.Tests;

/// <summary>
/// Drives *every initial option* of every event to a terminal state — not just the one greedy path
/// the resolve-sweep walks — with a <see cref="LogErrorSink"/> active, asserting the run swallows no
/// error-level log. This catches per-option faults the greedy sweep misses: an unguarded NRE on a
/// null UI singleton inside one option's effect (e.g. JungleMazeAdventure's "Safety in Numbers")
/// silently faults its fire-and-forget task, so the greedy walk — which takes a *different* option —
/// never sees it. Covers the index-0 act events and the shared (vote-based) events.
/// </summary>
public sealed class EventOptionSweepTests
{
    private readonly ITestOutputHelper _out;

    public EventOptionSweepTests(ITestOutputHelper output) => _out = output;

    public static IEnumerable<object[]> AllEvents =>
        Act1EventsTests.OvergrowthEvents
            .Concat(Act1EventsTests.UnderdocksEvents)
            .Concat(Act2EventsTests.HiveEvents)
            .Concat(Act3EventsTests.GloryEvents)
            .Concat(Act1EventsTests.SharedEvents)
            .Distinct(EventNameComparer.Instance);

    /// <summary>Resolve a canonical event by type name across every act variant plus the shared events.</summary>
    private static EventModel ResolveAnyEvent(string typeName) =>
        Enumerable.Range(0, ModelDb.ActsByIndex.Count)
            .SelectMany(i => ModelDb.ActsByIndex[i])
            .SelectMany(a => a.AllEvents)
            .Concat(ModelDb.AllSharedEvents)
            .First(e => e.GetType().Name == typeName);

    [Theory]
    [MemberData(nameof(AllEvents))]
    public async Task EveryOption_ResolvesWithoutSwallowedErrors(string eventName)
    {
        await Task.Run(() => RunEveryOption(eventName)).WaitAsync(TimeSpan.FromSeconds(180));
    }

    private void RunEveryOption(string eventName)
    {
        // First, enter once to discover this event's initial option indices.
        int[] optionIndices = WithFreshEvent(eventName, host =>
            host.ListOptions()
                .Where(o => o.Kind == OptionKind.ChooseEventOption)
                .Select(o => o.EventOptionIndex!.Value)
                .ToArray());

        if (optionIndices.Length == 0)
        {
            _out.WriteLine($"{eventName}: no actionable initial options (already terminal) — skipping");
            return;
        }

        foreach (int index in optionIndices)
        {
            using var errors = new LogErrorSink();
            WithFreshEvent(eventName, host =>
            {
                GameOption? opt = host.ListOptions()
                    .FirstOrDefault(o => o.Kind == OptionKind.ChooseEventOption && o.EventOptionIndex == index);
                if (opt is null)
                {
                    return 0; // option not offered this run (rng-gated); nothing to drive
                }
                host.Apply(opt);
                // Drive whatever the option spawned (sub-options, combat, rewards) to a terminal state.
                AutoPlayer.Advance(
                    host,
                    stop: s => s.Phase == GamePhase.Map || s.Phase == GamePhase.GameOver,
                    maxSteps: 3000);
                return 0;
            });

            IReadOnlyList<string> swallowed = errors.Errors;
            foreach (string e in swallowed)
            {
                _out.WriteLine($"{eventName} option {index} SWALLOWED ERROR: {e}");
            }
            Assert.True(swallowed.Count == 0,
                $"{eventName} option {index} swallowed {swallowed.Count} error(s); first: " +
                $"{(swallowed.Count > 0 ? swallowed[0].Split('\n')[0] : "")}");
        }
    }

    private static T WithFreshEvent<T>(string eventName, Func<GameHost, T> body)
    {
        GameHost host = TestNav.StartOnMap("TESTSEED");
        TestNav.SetHp(host, maxHp: 9999, currentHp: 9999); // survive any forced fight
        EventModel ev = ResolveAnyEvent(eventName);
        host.EnterEventDebug(ev);
        return body(host);
    }

    private sealed class EventNameComparer : IEqualityComparer<object[]>
    {
        public static readonly EventNameComparer Instance = new();
        public bool Equals(object[]? x, object[]? y) => (string)x![0] == (string)y![0];
        public int GetHashCode(object[] obj) => ((string)obj[0]).GetHashCode();
    }
}
