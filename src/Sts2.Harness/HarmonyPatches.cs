using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using HarmonyLib;
using MegaCrit.Sts2.Core.Localization;

namespace Sts2.Harness;

/// <summary>
/// Harmony patches that make the game tolerate the absence of the packed
/// localization tables (which live in the 1.9 GB .pck we don't ship). Display text
/// is irrelevant to mechanics, so missing tables/keys degrade to the key string
/// instead of throwing. This keeps logging and any incidental text formatting on the
/// game's hot paths from crashing the simulation.
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
}
