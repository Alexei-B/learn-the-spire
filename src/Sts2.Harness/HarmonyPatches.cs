using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using HarmonyLib;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;

namespace Sts2.Harness;

/// <summary>
/// Harmony patches that keep the headless harness running where the game assumes content we don't
/// ship or UI we don't build:
/// <list type="bullet">
/// <item>Missing packed localization tables (in the 1.9 GB .pck) degrade to the key string — display
/// text is irrelevant to mechanics — so logging/text formatting on hot paths doesn't crash.</item>
/// <item>The Crystal Sphere event minigame's UI screen (<c>NCrystalSphereScreen</c>, null headless)
/// is skipped and the plain-C# minigame is routed to the harness so it surfaces as agent choices.</item>
/// </list>
/// </summary>
internal static class HarmonyPatches
{
    private static readonly object Gate = new();
    private static bool _applied;

    private static readonly ConcurrentDictionary<string, LocTable> EmptyTables = new();

    public static void EnsureApplied()
    {
        if (_applied)
        {
            return;
        }
        lock (Gate)
        {
            if (_applied)
            {
                return;
            }

            var harmony = new Harmony("sts2.harness.localization");

            harmony.Patch(
                AccessTools.Method(typeof(LocManager), nameof(LocManager.GetTable)),
                finalizer: new HarmonyMethod(typeof(HarmonyPatches), nameof(GetTableFinalizer)));

            harmony.Patch(
                AccessTools.Method(typeof(LocTable), nameof(LocTable.GetRawText), new[] { typeof(string) }),
                finalizer: new HarmonyMethod(typeof(HarmonyPatches), nameof(GetRawTextFinalizer)));

            // Event option title/description lookups use LocString.GetIfExists, which returns null
            // for a missing key (our tables are empty). EventOption.AddLocVars then dereferences the
            // (null) description in CharacterModel.AddDetailsTo and NREs, faulting event init. Make
            // the lookups fall back to a key-named LocString (which renders as the key via the
            // patches above) so missing option text degrades instead of throwing.
            harmony.Patch(
                AccessTools.Method(typeof(EventModel), nameof(EventModel.GetOptionTitle)),
                postfix: new HarmonyMethod(typeof(HarmonyPatches), nameof(GetOptionTitlePostfix)));

            harmony.Patch(
                AccessTools.Method(typeof(EventModel), nameof(EventModel.GetOptionDescription)),
                postfix: new HarmonyMethod(typeof(HarmonyPatches), nameof(GetOptionDescriptionPostfix)));

            // The Crystal Sphere minigame's screen instantiates a UI scene (null headless) and pushes
            // it onto the overlay stack. Skip it and hand the live minigame to the harness, which
            // surfaces it as GamePhase.CrystalSphere and drives the cell-clicks the UI normally would.
            harmony.Patch(
                AccessTools.Method(
                    typeof(MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen),
                    nameof(MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen.ShowScreen)),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(ShowCrystalSphereScreenPrefix)));

            _applied = true;
        }
    }

    // If a loc table is missing, hand back an empty one named after the request so
    // callers keep working (and subsequent key lookups fall through to the key).
    private static Exception? GetTableFinalizer(Exception? __exception, string name, ref LocTable __result)
    {
        if (__exception != null)
        {
            __result = EmptyTables.GetOrAdd(name, n => new LocTable(n, new Dictionary<string, string>()));
        }
        return null;
    }

    // If a key is missing from a table, return the key itself rather than throwing.
    private static Exception? GetRawTextFinalizer(Exception? __exception, string key, ref string __result)
    {
        if (__exception != null)
        {
            __result = key;
        }
        return null;
    }

    // Missing event option title/description: hand back a key-named LocString (rendered as the key)
    // instead of null, so AddLocVars/AddDetailsTo don't NRE on it.
    private static void GetOptionTitlePostfix(EventModel __instance, string key, ref LocString? __result)
    {
        __result ??= new LocString(__instance.LocTable, key + ".title");
    }

    private static void GetOptionDescriptionPostfix(EventModel __instance, string key, ref LocString? __result)
    {
        __result ??= new LocString(__instance.LocTable, key + ".description");
    }

    // Skip the Crystal Sphere UI screen entirely (it would instantiate a null scene and NRE), routing
    // the live minigame to the active harness instead. Returning false suppresses the original; the
    // caller (CrystalSphereMinigame.PlayMinigame) discards the return value and then awaits the
    // minigame's own completion source, which the harness completes as the agent spends divinations.
    private static bool ShowCrystalSphereScreenPrefix(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame grid,
        ref MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen __result)
    {
        GameHost.CrystalSphereScreenHook?.Invoke(grid);
        __result = null!;
        return false;
    }
}
