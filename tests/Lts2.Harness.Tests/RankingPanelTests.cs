using System.Collections.Generic;
using System.Linq;
using Lts2.Tui;
using Xunit;

namespace Lts2.Harness.Tests;

/// <summary>
/// Unit tests for the TUI's pure agent-ranking formatter (<see cref="RankingPanel"/>) — the debug panel
/// that renders a decision engine's full scored ranking. Covers ordering (score descending), score
/// formatting, the top-pick highlight marker, graceful truncation, and the explicit decline / pending /
/// no-strategy states. The formatter is deliberately view-free so it needs no Terminal.Gui init.
/// </summary>
public sealed class RankingPanelTests
{
    // Flatten a rendered line to its plain text (dropping colour) for assertions.
    private static string Text(Line line) => string.Concat(line.Select(s => s.Text));

    private static List<string> Texts(IEnumerable<Line> lines) => lines.Select(Text).ToList();

    [Fact]
    public void Ranked_SortsByScoreDescending_MarksTopPick_AndFormatsScores()
    {
        var rows = new List<RankingPanel.Row>
        {
            new(5.0, "Play Defend", "Efficient block.", IsTopPick: false),
            new(9.25, "Play Strike -> Goblin", "Lethal on Goblin.", IsTopPick: true),
            new(1.0, "End turn", "End the turn.", IsTopPick: false),
        };

        List<Line> lines = RankingPanel.Format("Rules", RankingPanel.Status.Ranked, rows, 80);
        List<string> text = Texts(lines);

        // Header names the engine and the option count.
        Assert.Contains("Rules", text[0]);
        Assert.Contains("3 options", text[0]);

        // The option rows (skip header + blank) appear highest score first.
        List<string> optionRows = text.Where(t => t.Contains("Play ") || t.Contains("End turn")).ToList();
        int iStrike = optionRows.FindIndex(t => t.Contains("Play Strike"));
        int iDefend = optionRows.FindIndex(t => t.Contains("Play Defend"));
        int iEnd = optionRows.FindIndex(t => t.Contains("End turn"));
        Assert.True(iStrike < iDefend && iDefend < iEnd, $"expected Strike<Defend<End, got {iStrike},{iDefend},{iEnd}");

        // Score is rendered to two decimals, and the top pick (and only it) carries the ▸ marker.
        string strikeRow = optionRows[iStrike];
        Assert.Contains("9.25", strikeRow);
        Assert.StartsWith("▸", strikeRow);
        Assert.DoesNotContain("▸", optionRows[iDefend]);
        Assert.DoesNotContain("▸", optionRows[iEnd]);

        // The rationale is rendered (on its own indented line).
        Assert.Contains(text, t => t.Contains("Lethal on Goblin."));
    }

    [Fact]
    public void Ranked_TruncatesLongDescriptionAndRationale_ToWidth()
    {
        var rows = new List<RankingPanel.Row>
        {
            new(3.0,
                "Play an enormously long card description that will not remotely fit the panel width",
                "and an equally long rationale explaining exactly why this move is preferred right now",
                IsTopPick: true),
        };

        const int width = 24;
        List<string> text = Texts(RankingPanel.Format("Agent", RankingPanel.Status.Ranked, rows, width));

        Assert.All(text, t => Assert.True(t.Length <= width, $"line '{t}' exceeds width {width}"));
        // The overflowing description and rationale are both cut with an ellipsis.
        Assert.Contains(text, t => t.Contains("…"));
    }

    [Fact]
    public void Declined_SaysSo_Explicitly()
    {
        List<string> text = Texts(RankingPanel.Format(
            "Rules", RankingPanel.Status.Declined, System.Array.Empty<RankingPanel.Row>(), 60));

        Assert.Contains("Rules", text[0]);
        Assert.Contains(text, t => t.Contains("declined"));
    }

    [Fact]
    public void NoStrategy_And_Pending_HaveDistinctExplicitText()
    {
        List<string> none = Texts(RankingPanel.Format(
            "Rules", RankingPanel.Status.NoStrategy, System.Array.Empty<RankingPanel.Row>(), 60));
        Assert.Contains(none, t => t.Contains("no strategy"));

        List<string> pending = Texts(RankingPanel.Format(
            "Agent", RankingPanel.Status.Pending, System.Array.Empty<RankingPanel.Row>(), 60));
        Assert.Contains("Agent", pending[0]);
        Assert.Contains(pending, t => t.Contains("evaluating") || t.Contains("Waiting"));
    }

    [Fact]
    public void RowsFrom_MarksOnlyTheReferenceEqualTopPick()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        var engine = new RulesDecisionEngine();
        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();
        IReadOnlyList<ScoredOption> scored = engine.Evaluate(state, options);
        Assert.NotEmpty(scored);

        GameOption top = engine.Recommend(state, options)!;
        IReadOnlyList<RankingPanel.Row> rows = RankingPanel.RowsFrom(scored, top);

        Assert.Equal(scored.Count, rows.Count);
        Assert.Equal(1, rows.Count(r => r.IsTopPick));
        // The flagged row is the one whose option is the engine's pick.
        int topIdx = scored.ToList().FindIndex(s => ReferenceEquals(s.Option, top));
        Assert.True(rows[topIdx].IsTopPick);
    }
}
