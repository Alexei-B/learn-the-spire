using System;
using System.Linq;
using System.Text;
using MegaCrit.Sts2.Core.Models;
using Terminal.Gui;

namespace Sts2.Tui;

/// <summary>The choices made on the new-run screen.</summary>
internal sealed record RunConfig(CharacterModel Character, int Ascension, string Seed);

/// <summary>A modal dialog to configure a new run (character / ascension / seed).</summary>
internal static class NewRunDialog
{
    /// <summary>Show the dialog (modal). Returns the chosen config, or null if cancelled.</summary>
    public static RunConfig? Show()
    {
        var characters = ModelDb.AllCharacters.ToList();
        string[] names = characters.Select(c => c.Id.Entry).ToArray();

        var charLabel = new Label { X = 1, Y = 0, Text = "Choose your character:" };
        var radio = new RadioGroup { X = 2, Y = 1, RadioLabels = names };

        var ascLabel = new Label { X = 1, Y = Pos.Bottom(radio) + 1, Text = "Ascension (0-10):" };
        var ascField = new TextField { X = Pos.Right(ascLabel) + 1, Y = Pos.Top(ascLabel), Width = 6, Text = "0" };

        var seedLabel = new Label { X = 1, Y = Pos.Bottom(ascLabel) + 1, Text = "Seed:" };
        var seedField = new TextField { X = Pos.Right(seedLabel) + 1, Y = Pos.Top(seedLabel), Width = 24, Text = RandomSeed() };

        RunConfig? result = null;

        var dlg = new Dialog { Title = "New Run", Width = 58, Height = 16, ColorScheme = Theme.Base };

        var start = new Button { Text = "Start", IsDefault = true };
        start.Accepting += (_, e) =>
        {
            e.Cancel = true; // we handle it; don't let the command bubble further
            int asc = Math.Clamp(ParseInt(ascField.Text, 0), 0, 10);
            string seed = seedField.Text;
            if (string.IsNullOrWhiteSpace(seed))
            {
                seed = RandomSeed();
            }
            result = new RunConfig(characters[radio.SelectedItem], asc, seed.Trim());
            Application.RequestStop(dlg);
        };

        var cancel = new Button { Text = "Cancel" };
        cancel.Accepting += (_, e) =>
        {
            e.Cancel = true;
            result = null;
            Application.RequestStop(dlg);
        };

        dlg.Add(charLabel, radio, ascLabel, ascField, seedLabel, seedField);
        dlg.AddButton(start);
        dlg.AddButton(cancel);
        Application.Run(dlg);
        dlg.Dispose();
        return result;
    }

    private static int ParseInt(string? s, int fallback) =>
        int.TryParse(s, out int v) ? v : fallback;

    private static string RandomSeed()
    {
        const string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        var rng = Random.Shared;
        var sb = new StringBuilder(8);
        for (int i = 0; i < 8; i++)
        {
            sb.Append(alphabet[rng.Next(alphabet.Length)]);
        }
        return sb.ToString();
    }
}
