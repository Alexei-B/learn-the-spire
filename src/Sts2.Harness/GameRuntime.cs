using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2.Harness;

/// <summary>
/// One-time, process-wide initialization of the game's static systems for headless
/// play. Mirrors the logic half of the game's <c>OneTimeInitialization</c>
/// (ExecuteVeryEarly + ExecuteEssential) while skipping every UI/asset step
/// (atlases, resource-format loaders, etc.).
/// </summary>
public static class GameRuntime
{
    private static readonly object Gate = new();
    private static bool _initialized;

    public static void EnsureInitialized()
    {
        if (_initialized)
        {
            return;
        }

        lock (Gate)
        {
            if (_initialized)
            {
                return;
            }

            // Headless flags: in-memory saves, no animations/delays/UI waits.
            TestMode.IsOn = true;
            NonInteractiveMode.AutoSlayerCheck = () => true;

            // Make missing localization (the tables live only in the .pck) degrade to
            // returning the key, so logging / incidental text formatting never throws.
            HarmonyPatches.EnsureApplied();

            // --- ExecuteVeryEarly (logic only) ---
            SaveManager.Instance.InitSettingsDataForTest();
            // A profile id is required before any progress save; use slot 0.
            SaveManager.Instance.InitProfileId(0);
            // Disable first-time-user-experience tutorials: their Create() methods load
            // Godot PackedScenes. Disabled => SeenFtue() returns true => those branches
            // (e.g. the combat-rules popup) are skipped.
            SaveManager.Instance.SetFtuesEnabled(false);
            // In TestMode this short-circuits to ModManagerState.Skipped without
            // touching the (Godot) mod filesystem, which is all we need: it just
            // has to leave State != None so ReflectionHelper.ModTypes is usable.
            ModManager.Initialize(null!, null, null).GetAwaiter().GetResult();

            // --- ExecuteEssential (logic only; atlases/ResourceLoader skipped) ---
            // LocManager needs a language-completion file to initialize; the real one
            // is in the .pck. Write a minimal stand-in so Instance becomes non-null
            // (many logic paths call LocManager.Instance). Tables stay empty and the
            // Harmony patches above make lookups return their keys.
            EnsureMinimalLocalizationData();
            LocManager.Initialize();
            ModelDb.Init();
            ModelIdSerializationCache.Init();
            ModelDb.InitIds();
            MessageTypes.Initialize();
            ActionTypes.Initialize();

            _initialized = true;
        }
    }

    /// <summary>
    /// Write the minimal localization-completion file LocManager.Initialize() requires.
    /// The path resolves through the shim's globalized temp filesystem.
    /// </summary>
    private static void EnsureMinimalLocalizationData()
    {
        string completionPath = Godot.ProjectSettings.GlobalizePath("res://localization/completion.json");
        System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(completionPath)!);
        System.IO.File.WriteAllText(completionPath, "{\"eng\":1}");
    }
}
