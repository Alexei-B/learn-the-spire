using System.Collections.Generic;
using Terminal.Gui;

namespace Lts2.Tui;

/// <summary>Modal dialogs for the save/load menu.</summary>
internal static class SaveDialogs
{
    /// <summary>Prompt for a save-slot name. Returns the entered name, or null if cancelled.</summary>
    public static string? PromptSaveName(string defaultName)
    {
        var label = new Label { X = 1, Y = 0, Text = "Save name:" };
        var field = new TextField { X = 1, Y = 1, Width = Dim.Fill(1), Text = defaultName };
        string? result = null;

        var dlg = new Dialog { Title = "Save Run", Width = 50, Height = 8, ColorScheme = Theme.Base };
        var save = new Button { Text = "Save", IsDefault = true };
        save.Accepting += (_, e) =>
        {
            e.Cancel = true;
            result = field.Text;
            Application.RequestStop(dlg);
        };
        var cancel = new Button { Text = "Cancel" };
        cancel.Accepting += (_, e) => { e.Cancel = true; result = null; Application.RequestStop(dlg); };

        dlg.Add(label, field);
        dlg.AddButton(save);
        dlg.AddButton(cancel);
        Application.Run(dlg);
        dlg.Dispose();
        return result;
    }

    /// <summary>Pick a save to load. Returns the chosen file path, or null if none/cancelled.</summary>
    public static string? PromptLoad()
    {
        IReadOnlyList<SaveStore.SaveInfo> saves = SaveStore.List();
        if (saves.Count == 0)
        {
            MessageBox.Query("Load Run", "\nNo saved runs found.\n", "OK");
            return null;
        }

        var labels = new string[saves.Count];
        for (int i = 0; i < saves.Count; i++)
        {
            labels[i] = saves[i].Describe();
        }

        var radio = new RadioGroup { X = 1, Y = 1, Width = Dim.Fill(1), RadioLabels = labels };
        var heading = new Label { X = 1, Y = 0, Text = "Choose a run to load:" };
        string? result = null;

        int height = System.Math.Min(saves.Count + 6, 22);
        var dlg = new Dialog { Title = "Load Run", Width = 90, Height = height, ColorScheme = Theme.Base };
        var load = new Button { Text = "Load", IsDefault = true };
        load.Accepting += (_, e) =>
        {
            e.Cancel = true;
            result = saves[radio.SelectedItem].Path;
            Application.RequestStop(dlg);
        };
        var cancel = new Button { Text = "Cancel" };
        cancel.Accepting += (_, e) => { e.Cancel = true; result = null; Application.RequestStop(dlg); };

        dlg.Add(heading, radio);
        dlg.AddButton(load);
        dlg.AddButton(cancel);
        Application.Run(dlg);
        dlg.Dispose();
        return result;
    }
}
