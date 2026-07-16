using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Lts2.Harness;

namespace Lts2.Tui;

/// <summary>
/// Pure formatter for the agent-ranking debug panel: turns a decision engine's scored options (protocol
/// v1's per-option <c>score</c>/<c>rationale</c>) into coloured <see cref="Line"/>s — the full ranking
/// sorted by score descending, the Tab auto-play pick marked, and explicit text for the still-evaluating,
/// declined and no-strategy states. Deliberately free of Terminal.Gui view plumbing and live game objects
/// (it works over the plain <see cref="Row"/> record) so its ordering / truncation / decline / highlight
/// behaviour is covered by a straight unit test — see <c>RankingPanelTests</c>.
/// </summary>
internal static class RankingPanel
{
    /// <summary>What the panel has to show for the current decision point.</summary>
    internal enum Status
    {
        /// <summary>No run / nothing to rank here — the panel shows a neutral placeholder.</summary>
        NoStrategy,

        /// <summary>The engine is still being asked (async; an external agent can take seconds).</summary>
        Pending,

        /// <summary>The engine returned an empty ranking — it declined to recommend a move.</summary>
        Declined,

        /// <summary>The engine returned a ranking to display.</summary>
        Ranked,
    }

    /// <summary>One ranked option, decoupled from the live <see cref="ScoredOption"/> so the formatter is
    /// trivially testable. <paramref name="IsTopPick"/> flags the option Tab would apply (the engine's
    /// <c>Best</c>, tie-broken toward the earliest option) so the marker matches auto-play exactly.</summary>
    internal sealed record Row(double Score, string Description, string? Rationale, bool IsTopPick);

    /// <summary>Build panel rows from a completed evaluation, flagging the option Tab would apply.</summary>
    internal static IReadOnlyList<Row> RowsFrom(IReadOnlyList<ScoredOption> scored, GameOption? topPick)
    {
        var rows = new List<Row>(scored.Count);
        foreach (ScoredOption s in scored)
        {
            rows.Add(new Row(s.Score, s.Option.Description, s.Rationale, ReferenceEquals(s.Option, topPick)));
        }
        return rows;
    }

    /// <summary>
    /// The coloured panel body for <paramref name="engineName"/> and <paramref name="status"/>. For
    /// <see cref="Status.Ranked"/> the <paramref name="rows"/> are sorted by score descending (stable, so
    /// ties keep their input order) and rendered as "rank · score · description" with an indented, dimmed
    /// rationale beneath; the top pick is marked "▸" in teal. All text is truncated to fit
    /// <paramref name="width"/> columns.
    /// </summary>
    internal static List<Line> Format(string engineName, Status status, IReadOnlyList<Row> rows, int width)
    {
        width = Math.Max(8, width);
        var lines = new List<Line>();

        string header = status switch
        {
            Status.Pending => $"{engineName} · evaluating…",
            Status.Declined => $"{engineName} · declined",
            Status.Ranked => $"{engineName} · {rows.Count} option{(rows.Count == 1 ? "" : "s")}",
            _ => "no strategy",
        };
        lines.Add(new Line().Add(Truncate(header, width), Theme.Teal));
        lines.Add(new Line());

        switch (status)
        {
            case Status.NoStrategy:
                lines.Add(new Line().Dim(Truncate("Nothing to rank right now.", width)));
                return lines;
            case Status.Pending:
                lines.Add(new Line().Dim(Truncate("Waiting for the agent…", width)));
                return lines;
            case Status.Declined:
                lines.Add(new Line().Dim(Truncate("The agent declined to recommend a move here.", width)));
                return lines;
        }

        // Ranked: stable sort by score descending (ties keep input order, matching Best's tie-break).
        List<Row> ordered = rows
            .Select((r, i) => (row: r, i))
            .OrderByDescending(t => t.row.Score)
            .ThenBy(t => t.i)
            .Select(t => t.row)
            .ToList();

        for (int i = 0; i < ordered.Count; i++)
        {
            Row row = ordered[i];
            string marker = row.IsTopPick ? "▸ " : "  ";
            string rank = (i + 1).ToString(CultureInfo.InvariantCulture) + ". ";
            string score = row.Score.ToString("0.00", CultureInfo.InvariantCulture) + "  ";
            int used = marker.Length + rank.Length + score.Length;

            var line = new Line();
            line.Add(marker, row.IsTopPick ? Theme.Teal : Theme.Dim);
            line.Add(rank, Theme.Dim);
            line.Add(score, Theme.Gold);
            line.Add(Truncate(row.Description, Math.Max(1, width - used)), row.IsTopPick ? Theme.Teal : Theme.Fg);
            lines.Add(line);

            if (!string.IsNullOrWhiteSpace(row.Rationale))
            {
                lines.Add(new Line().Dim("     " + Truncate(row.Rationale!.Trim(), Math.Max(1, width - 5))));
            }
        }
        return lines;
    }

    /// <summary>Collapse newlines and hard-truncate <paramref name="s"/> to <paramref name="width"/>
    /// columns, marking a cut with an ellipsis.</summary>
    private static string Truncate(string s, int width)
    {
        s = s.Replace('\n', ' ').Replace('\r', ' ');
        if (s.Length <= width)
        {
            return s;
        }
        return width <= 1 ? s.Substring(0, width) : s.Substring(0, width - 1) + "…";
    }
}
