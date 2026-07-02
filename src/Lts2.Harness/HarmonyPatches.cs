using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes;

namespace Lts2.Harness;

/// <summary>
/// Harmony patches that keep the headless harness running where the game assumes content we don't
/// ship or UI we don't build:
/// <list type="bullet">
/// <item>Missing packed localization tables (in the 1.9 GB .pck) degrade to the key string — display
/// text is irrelevant to mechanics — so logging/text formatting on hot paths doesn't crash.</item>
/// <item>The Crystal Sphere event minigame's UI screen (<c>NCrystalSphereScreen</c>, null headless)
/// is skipped and the plain-C# minigame is routed to the harness so it surfaces as agent choices.</item>
/// <item>The PunchOff event's cosmetic <c>NGame.Instance.ScreenShakeTrauma</c> call — an unguarded
/// <c>callvirt</c> on the null UI singleton — is removed from its option's IL so the option's logic
/// (curse, relic reward, finish) runs instead of NRE'ing at the call site.</item>
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

            // CardModel.SelectionScreenPrompt *throws* "No selection screen prompt for X" when its
            // loc key is missing (our tables are empty), rather than degrading like other lookups.
            // Cards that raise a mid-effect card selection read it (e.g. Wish's draw-pile pick), so a
            // missing prompt faults the card mid-play. Swallow the throw and hand back a key-named
            // LocString (rendered as the key) — the prompt is display-only, irrelevant to mechanics.
            harmony.Patch(
                AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Models.CardModel), "SelectionScreenPrompt"),
                finalizer: new HarmonyMethod(typeof(HarmonyPatches), nameof(SelectionScreenPromptFinalizer)));

            // The Crystal Sphere minigame's screen instantiates a UI scene (null headless) and pushes
            // it onto the overlay stack. Skip it and hand the live minigame to the harness, which
            // surfaces it as GamePhase.CrystalSphere and drives the cell-clicks the UI normally would.
            harmony.Patch(
                AccessTools.Method(
                    typeof(MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen),
                    nameof(MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen.ShowScreen)),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(ShowCrystalSphereScreenPrefix)));

            // CardSelectCmd.FromChooseABundleScreen (ScrollBoxes' "pick one of two card bundles") shows a
            // UI screen that is null headless and, worse, silently auto-takes bundles[0] under TestMode.
            // Route it to the active harness so the bundles surface as GamePhase.BundleChoice options.
            harmony.Patch(
                AccessTools.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseABundleScreen)),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(FromChooseABundleScreenPrefix)));

            // Several event options call NGame.Instance.ScreenShakeTrauma — a callvirt on the null
            // headless UI singleton that NREs *before* the option's actual effect (relic reward,
            // SetEventFinished, …), so the event can't resolve. Making NGame.Instance non-null would
            // defeat the hundreds of NGame.Instance?.… guards the logic relies on, so instead strip
            // just this cosmetic call from each affected option's IL (its async state machine's
            // MoveNext). PunchOff's "Nab" and Amalgamator's two combine options each carry one.
            StripScreenShake(harmony, typeof(PunchOff), "Nab");
            StripScreenShake(harmony, typeof(Amalgamator), "CombineStrikes", "CombineDefends");

            ApplyKaiserCrabBackgroundPatches(harmony);

            // SoulNexus (an act-3 elite) sets a death animation in its Died handler via
            // NCombatRoom.Instance.GetCreatureNode(…) — an unguarded callvirt on the null headless
            // combat-room UI singleton, which NREs the moment the creature dies, faulting the kill.
            // The handler is purely cosmetic (the only other line just unsubscribes itself, harmless
            // to skip since the creature dies once), so no-op it. (Several other monsters reach for
            // NCombatRoom.Instance.GetCreatureNode too, but only on paths never hit headless.)
            harmony.Patch(
                AccessTools.Method(
                    typeof(MegaCrit.Sts2.Core.Models.Monsters.SoulNexus), "AfterDeath",
                    new[] { typeof(MegaCrit.Sts2.Core.Entities.Creatures.Creature) }),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipVoidPrefix)));

            // The Decimillipede elite's segments revive each other until all are dead at once; when the
            // last one dies, ReattachPower.DoFadeOutOnAllSegments fades out every segment's *UI node*.
            // Headless it builds a node list off the (present-but-inert) combat room and calls Godot
            // Node methods the shim doesn't implement (Node.GetIndex(bool)), throwing mid-kill and
            // aborting the win-condition check — so the fully-dead Decimillipede never dies. The method
            // is purely visual (the segments are already logically dead, so IsEnding is already true);
            // no-op it and the kill completes, ending the fight.
            harmony.Patch(
                AccessTools.Method(
                    typeof(MegaCrit.Sts2.Core.Models.Powers.ReattachPower), "DoFadeOutOnAllSegments"),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipVoidPrefix)));

            ApplyTrialEventPatches(harmony);

            // NAudioManager.Instance is NGame.Instance?.AudioManager — null headless. SfxCmd guards
            // its audio calls on NonInteractiveMode, but a handful of sites dereference the singleton
            // *unguarded* (monster death SFX in BeforeDeath, game-over music in CreatureCmd), which
            // NREs on the null instance. Every playback method early-returns on TestMode.IsOn (true
            // here), so handing the getter an inert instance makes those calls no-op — no audio plays.
            harmony.Patch(
                AccessTools.PropertyGetter(
                    typeof(MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager), "Instance"),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(AudioManagerInstancePrefix)));

            ApplyCosmeticUiStripPatches(harmony);

            _applied = true;
        }
    }

    // The KaiserCrab boss (Crusher) reaches into a UI background node for all its visuals: its
    // Background getter resolves NCombatRoom.Instance.Background (or NBestiary's layout), both null
    // headless, so it *throws* — and it is touched on creature setup (AfterAddedToRoom), on every HP
    // change, on death, and on every move. None of it is mechanical. So we hand the getter a single
    // inert NKaiserCrabBossBackground (constructor skipped, never added to a scene) and no-op that
    // type's cosmetic anim methods (which would otherwise NRE on the uninitialized animation nodes),
    // leaving the move/damage/death *logic* — which runs after these calls — to execute normally.
    private static readonly object _crabBgGate = new();
    private static MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground? _inertCrabBg;

    private static void ApplyKaiserCrabBackgroundPatches(Harmony harmony)
    {
        // The KaiserCrab boss is two monsters — Crusher (left/body) and Rocket (right arm) — each
        // with its own throwing Background getter onto the same UI node. Patch both to the inert one.
        foreach (Type monster in new[]
                 {
                     typeof(MegaCrit.Sts2.Core.Models.Monsters.Crusher),
                     typeof(MegaCrit.Sts2.Core.Models.Monsters.Rocket),
                 })
        {
            harmony.Patch(
                AccessTools.PropertyGetter(monster, "Background"),
                prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(CrusherBackgroundPrefix)));
        }

        // No-op the node's animation-playback methods (PlayAttackAnim/PlayHurtAnim/PlayArmDeathAnim/
        // PlayBodyDeathAnim/PlayRight…) — the only ones Crusher calls, and all purely visual (they
        // touch live Godot animation state we never build). We match by the "Play" prefix so we do
        // NOT touch the source-generated marshalling overrides (InvokeGodotClassMethod, _Ready, …),
        // which reference native-interop types the shim deliberately omits. SetVisible et al. live on
        // the shim's CanvasItem and work on the inert instance as-is.
        Type bg = typeof(MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground);
        foreach (MethodInfo m in bg.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Instance
                     | BindingFlags.Public | BindingFlags.NonPublic))
        {
            if (!m.Name.StartsWith("Play", StringComparison.Ordinal)
                || m.IsAbstract || m.IsGenericMethodDefinition || m.GetMethodBody() is null)
            {
                continue;
            }
            if (m.ReturnType == typeof(void))
            {
                harmony.Patch(m, prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipVoidPrefix)));
            }
            else if (typeof(System.Threading.Tasks.Task).IsAssignableFrom(m.ReturnType))
            {
                harmony.Patch(m, prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipTaskPrefix)));
            }
        }
    }

    private static bool CrusherBackgroundPrefix(
        ref MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground __result)
    {
        lock (_crabBgGate)
        {
            _inertCrabBg ??= (MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground)
                RuntimeHelpers.GetUninitializedObject(
                    typeof(MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground));
        }
        __result = _inertCrabBg;
        return false;
    }

    private static readonly object _audioGate = new();
    private static MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager? _inertAudio;

    private static bool AudioManagerInstancePrefix(
        ref MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager __result)
    {
        lock (_audioGate)
        {
            _inertAudio ??= (MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager)
                RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager));
        }
        __result = _inertAudio;
        return false;
    }

    // Several events run cosmetic UI calls inside an `if (LocalContext.IsMe(owner))` block — which is
    // TRUE for the local player headless — that NRE on null UI singletons: NDebugAudioManager.Instance
    // (= NGame.Instance?.DebugAudio, null) for temporary SFX, and NGame.Instance for screen rumble. The
    // shared JungleMazeAdventure ("Safety in Numbers") and DenseVegetation ("Rest") options both fault
    // their fire-and-forget effect task this way *before* the mechanical payout (gold/heal/finish). We
    // can't no-op the audio methods directly (Harmony can't read NDebugAudioManager.Play's body — it
    // references Godot.AudioStream, absent from the shim) nor make NGame.Instance non-null (hundreds of
    // `?.` guards rely on it). So, like the existing ScreenShakeTrauma strip, we transpile the cosmetic
    // *calls* out of the option state machines: the (null) receiver and args are popped and a default
    // return value pushed, dropping the call while every mechanical instruction runs untouched.
    private static void ApplyCosmeticUiStripPatches(Harmony harmony)
    {
        StripCosmeticUi(harmony, typeof(MegaCrit.Sts2.Core.Models.Events.JungleMazeAdventure),
            "SafetyInNumbers", "DontNeedHelp");
        StripCosmeticUi(harmony, typeof(MegaCrit.Sts2.Core.Models.Events.DenseVegetation),
            "TrudgeOn", "Rest");
    }

    // The cosmetic UI calls we strip: NDebugAudioManager's playback methods and NGame's screen-shake.
    private static readonly HashSet<MethodBase> _cosmeticCalls = new()
    {
        AccessTools.Method(typeof(MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager), "Play"),
        AccessTools.Method(typeof(MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager), "Stop"),
        AccessTools.Method(typeof(MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager), "StopAll"),
        AccessTools.Method(typeof(NGame), nameof(NGame.ScreenRumble)),
    };

    // Patch the compiler-generated MoveNext of each named async option method (matched by name
    // fragment among the type's nested IAsyncStateMachine types) with the cosmetic-call-stripping
    // transpiler. Mirrors StripScreenShake.
    private static void StripCosmeticUi(Harmony harmony, Type containing, params string[] methodNameFragments)
    {
        Type[] nested = containing.GetNestedTypes(BindingFlags.NonPublic | BindingFlags.Public);
        foreach (string fragment in methodNameFragments)
        {
            Type stateMachine = nested.First(t =>
                t.Name.Contains(fragment) && typeof(IAsyncStateMachine).IsAssignableFrom(t));
            harmony.Patch(
                AccessTools.Method(stateMachine, "MoveNext"),
                transpiler: new HarmonyMethod(typeof(HarmonyPatches), nameof(StripCosmeticUiTranspiler)));
        }
    }

    // Drop each call to a cosmetic UI method: pop its arguments and (null) receiver to keep the stack
    // balanced, and — for the one that returns a value (NDebugAudioManager.Play → int) — push a default
    // in its place. Every other instruction (including the receiver-producing get_Instance, which on a
    // null-conditional just yields null) is left exactly as is.
    private static IEnumerable<CodeInstruction> StripCosmeticUiTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        foreach (CodeInstruction instruction in instructions)
        {
            if (instruction.operand is MethodInfo method && _cosmeticCalls.Contains(method))
            {
                int pops = method.GetParameters().Length + 1; // arguments + the receiver
                for (int i = 0; i < pops; i++)
                {
                    var pop = new CodeInstruction(OpCodes.Pop);
                    if (i == 0)
                    {
                        // Carry the original call's branch labels / exception blocks onto the first pop.
                        pop.labels.AddRange(instruction.labels);
                        pop.blocks.AddRange(instruction.blocks);
                    }
                    yield return pop;
                }
                if (method.ReturnType == typeof(int))
                {
                    yield return new CodeInstruction(OpCodes.Ldc_I4_0);
                }
                else if (method.ReturnType != typeof(void))
                {
                    throw new InvalidOperationException(
                        $"StripCosmeticUiTranspiler can't supply a default for {method.ReturnType} ({method}).");
                }
            }
            else
            {
                yield return instruction;
            }
        }
    }

    // The Trial event's Accept option drives the event through the event-room portrait UI:
    // NEventRoom.Instance.Layout.RemoveNodesOnPortrait()/SetPortrait()/AddVfxAnchoredToPortrait() — all
    // unguarded callvirts on the null headless NEventRoom singleton (and a scene instantiate), so Accept
    // NREs before it builds the verdict sub-options (curses/relics/rewards/card-selects) that are the
    // event's actual content. NEventRoom.Instance is reached unguarded *only* here (everything else uses
    // the null-safe `?.VfxContainer`, and nothing branches on it being null), so we hand it — and its
    // Layout — single inert instances and no-op the cosmetic portrait methods, leaving Accept's option
    // generation to run. (Constructors skipped; never added to a scene.)
    private static readonly object _eventRoomGate = new();
    private static MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom? _inertEventRoom;
    private static MegaCrit.Sts2.Core.Nodes.Events.NEventLayout? _inertEventLayout;

    private static void ApplyTrialEventPatches(Harmony harmony)
    {
        harmony.Patch(
            AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom), "Instance"),
            prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(EventRoomInstancePrefix)));
        harmony.Patch(
            AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom), "Layout"),
            prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(EventRoomLayoutPrefix)));
        // SetPortrait(NEventRoom) → Layout.SetPortrait; no-op it directly. RemoveNodesOnPortrait is on
        // the layout. AddVfxAnchoredToPortrait is no-op'd at the Trial level (it also instantiates a
        // scene before the layout call), which covers the only path that reaches the layout's vfx add.
        harmony.Patch(
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom), "SetPortrait"),
            prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipVoidPrefix)));
        harmony.Patch(
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.Events.NEventLayout), "RemoveNodesOnPortrait"),
            prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipVoidPrefix)));
        harmony.Patch(
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.Events.Trial), "AddVfxAnchoredToPortrait"),
            prefix: new HarmonyMethod(typeof(HarmonyPatches), nameof(SkipVoidPrefix)));
    }

    private static bool EventRoomInstancePrefix(ref MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom __result)
    {
        lock (_eventRoomGate)
        {
            _inertEventRoom ??= (MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom)
                RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom));
        }
        __result = _inertEventRoom;
        return false;
    }

    private static bool EventRoomLayoutPrefix(ref MegaCrit.Sts2.Core.Nodes.Events.NEventLayout __result)
    {
        lock (_eventRoomGate)
        {
            _inertEventLayout ??= (MegaCrit.Sts2.Core.Nodes.Events.NEventLayout)
                RuntimeHelpers.GetUninitializedObject(typeof(MegaCrit.Sts2.Core.Nodes.Events.NEventLayout));
        }
        __result = _inertEventLayout;
        return false;
    }

    private static bool SkipVoidPrefix() => false;

    private static bool SkipTaskPrefix(ref System.Threading.Tasks.Task __result)
    {
        __result = System.Threading.Tasks.Task.CompletedTask;
        return false;
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

    // CardModel.SelectionScreenPrompt throws when its loc key is missing; on that throw, return a
    // key-named LocString (which renders as the key via the table/key patches) so the card's
    // mid-effect selection proceeds with placeholder prompt text instead of faulting.
    private static Exception? SelectionScreenPromptFinalizer(
        Exception? __exception, MegaCrit.Sts2.Core.Models.CardModel __instance, ref LocString __result)
    {
        if (__exception != null)
        {
            __result = new LocString("cards", __instance.Id.Entry + ".selectionScreenPrompt");
        }
        return null;
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

    // Route the "choose a bundle" selection (ScrollBoxes) to the active harness instead of the null UI
    // screen / TestMode's silent bundles[0] auto-pick. The prefix replaces the async method's result
    // with the harness's task, which completes with the chosen bundle once the agent picks one. With no
    // harness installed, fall through to the original so nothing else that reaches this method breaks.
    private static bool FromChooseABundleScreenPrefix(
        Player player,
        IReadOnlyList<IReadOnlyList<CardModel>> bundles,
        ref Task<IEnumerable<CardModel>> __result)
    {
        var hook = GameHost.BundleChoiceHook;
        if (hook is null)
        {
            return true;
        }
        __result = hook(player, bundles);
        return false;
    }

    // Strip the cosmetic NGame.Instance.ScreenShakeTrauma call from each named async option's
    // compiler-generated state machine (the option methods are `async Task`, so the call lives in a
    // nested IAsyncStateMachine's MoveNext, matched by the option method name).
    private static void StripScreenShake(Harmony harmony, Type containing, params string[] methodNameFragments)
    {
        Type[] nested = containing.GetNestedTypes(BindingFlags.NonPublic | BindingFlags.Public);
        foreach (string fragment in methodNameFragments)
        {
            Type stateMachine = nested.First(t =>
                t.Name.Contains(fragment) && typeof(IAsyncStateMachine).IsAssignableFrom(t));
            harmony.Patch(
                AccessTools.Method(stateMachine, "MoveNext"),
                transpiler: new HarmonyMethod(typeof(HarmonyPatches), nameof(StripScreenShakeTranspiler)));
        }
    }

    // Replace `NGame.Instance.ScreenShakeTrauma(strength)` with two pops: the receiver
    // (null NGame.Instance) and the strength arg are discarded, keeping the stack balanced, so the
    // call is dropped without an NRE while every other instruction (including the receiver-producing
    // get_Instance and the surrounding option logic) is left exactly as is.
    private static IEnumerable<CodeInstruction> StripScreenShakeTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        MethodInfo screenShake = AccessTools.Method(typeof(NGame), nameof(NGame.ScreenShakeTrauma));
        foreach (CodeInstruction instruction in instructions)
        {
            if (instruction.operand is MethodInfo method && method == screenShake)
            {
                var popArg = new CodeInstruction(OpCodes.Pop);
                popArg.labels.AddRange(instruction.labels);
                popArg.blocks.AddRange(instruction.blocks);
                yield return popArg;            // discard the strength argument
                yield return new CodeInstruction(OpCodes.Pop); // discard the (null) NGame.Instance receiver
            }
            else
            {
                yield return instruction;
            }
        }
    }
}
