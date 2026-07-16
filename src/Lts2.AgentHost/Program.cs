using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Lts2.Agent;
using Lts2.Agent.Wire;
using Lts2.Harness;

namespace Lts2.AgentHost;

/// <summary>
/// Headless entry point for the training environment. A remote driver (e.g. a Python trainer) spawns
/// this process and speaks the line protocol over its stdio: one JSON command per line in,
/// one JSON observation per line out. Keeps stdout reserved strictly for protocol messages by routing
/// all game logging to stderr.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        // Protocol messages are UTF-8 JSON lines; make stdio agree regardless of the console default.
        var utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        Console.OutputEncoding = utf8NoBom;
        Console.InputEncoding = utf8NoBom;

        // The game logs verbosely via Godot.GD.*; send it to stderr so stdout stays clean JSON lines.
        Godot.GD.Out = Console.Error;
        Godot.GD.Err = Console.Error;

        // Diagnostic: run only full-reset scenario builds in a tight loop, so a profiler (dotnet-trace)
        // can flame-graph exactly where a reset (GenerateMap etc.) spends its time in one clean process.
        if (args.Length > 0 && args[0] == "--bench-reset")
        {
            int n = args.Length > 1 && int.TryParse(args[1], out int x) ? x : 300;
            // Initialize (Harmony patches + ModelDb) up front, exactly as the server does, before anything
            // touches ModelDb — otherwise a lazy ModelDb.Init runs without the patches and fails.
            Lts2.Harness.GameRuntime.EnsureInitialized();
            var rng = new Random(1);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < n; i++)
            {
                Lts2.Harness.CombatScenario.Create($"B{i}", rng, "NECROBINDER", 0.15, 0.05, useStarterDeck: true, act: 0);
            }
            Console.Error.WriteLine($"[bench-reset] {n} resets in {sw.ElapsedMilliseconds}ms ({sw.ElapsedMilliseconds / (double)n:0.0}ms each)");
            return 0;
        }

        // Dump the static per-card metadata catalog to stdout (JSON) so the Python trainer can build its
        // static feature table + a stable per-card embedding index. Emits, per card: the CardTags,
        // the canonical CardKeywords, and the declared dynamic-var keys — the rich semantics the per-step
        // observation drops. Run once: `Lts2.AgentHost --dump-cards > cards.json`.
        if (args.Length > 0 && args[0] == "--dump-cards")
        {
            Lts2.Harness.GameRuntime.EnsureInitialized();
            var rows = new System.Collections.Generic.List<object>();
            foreach (var c in System.Linq.Enumerable.OrderBy(
                         MegaCrit.Sts2.Core.Models.ModelDb.AllCards, c => c.Id.Entry, System.StringComparer.Ordinal))
            {
                string[] Safe(System.Func<System.Collections.Generic.IEnumerable<string>> f)
                {
                    try { return System.Linq.Enumerable.ToArray(f()); } catch { return System.Array.Empty<string>(); }
                }
                CardCatalog.PoolCategory category = CardCatalog.CategoryOf(c);
                rows.Add(new
                {
                    id = c.Id.Entry,
                    type = c.Type.ToString(),
                    rarity = c.Rarity.ToString(),
                    // Pool membership the realistic sampler and the deck-distribution report need. `pool` is
                    // the card's home-pool title (character name / "colorless" / "curse" / "status" / …);
                    // `category` is the coarse class; the flags are the ones the tokenizer keys off.
                    pool = CardCatalog.PoolTitle(c),
                    category = category.ToString(),
                    colorless = category == CardCatalog.PoolCategory.Colorless,
                    curse = category == CardCatalog.PoolCategory.Curse,
                    status = category == CardCatalog.PoolCategory.Status,
                    tags = Safe(() => System.Linq.Enumerable.Select(c.Tags, t => t.ToString())),
                    keywords = Safe(() => System.Linq.Enumerable.Select(c.CanonicalKeywords, k => k.ToString())),
                    varKeys = Safe(() => c.DynamicVars.Keys),
                });
            }
            Console.Out.Write(System.Text.Json.JsonSerializer.Serialize(rows, AgentJson.Options));
            Console.Out.Flush();
            return 0;
        }

        // Dump the static per-power metadata catalog to stdout (JSON), mirroring --dump-cards, so the Python
        // trainer can build a stable per-power embedding index + static feature table. Emits, per power: its
        // id, PowerType (Buff/Debuff), stack/instance type, and the negative-amount flag — the cheap static
        // metadata the model type exposes. Run once: `Lts2.AgentHost --dump-powers > powers.json`.
        if (args.Length > 0 && args[0] == "--dump-powers")
        {
            Lts2.Harness.GameRuntime.EnsureInitialized();
            var rows = new System.Collections.Generic.List<object>();
            foreach (var p in System.Linq.Enumerable.OrderBy(
                         MegaCrit.Sts2.Core.Models.ModelDb.AllPowers, p => p.Id.Entry, System.StringComparer.Ordinal))
            {
                string[] SafeVarKeys()
                {
                    try { return System.Linq.Enumerable.ToArray(p.DynamicVars.Keys); }
                    catch { return System.Array.Empty<string>(); }
                }
                rows.Add(new
                {
                    id = p.Id.Entry,
                    type = p.Type.ToString(),
                    stackType = p.StackType.ToString(),
                    instanceType = p.InstanceType.ToString(),
                    allowNegative = p.AllowNegative,
                    varKeys = SafeVarKeys(),
                });
            }
            Console.Out.Write(System.Text.Json.JsonSerializer.Serialize(rows, AgentJson.Options));
            Console.Out.Flush();
            return 0;
        }

        // Fuzz combat to surface the intermittent errors/timeouts that random-character/random-encounter
        // rollouts hit (reward-sync, NRE, EndTurn timeouts): build many random scenarios and play each to
        // combat end with a random legal policy, catching every failure with its full stack + the
        // (character/encounter) context. Deterministic (fixed seed) so a failing case can be re-run.
        // Usage: `Lts2.AgentHost --fuzz-fights 300`.
        if (args.Length > 0 && args[0] == "--fuzz-fights")
        {
            int n = args.Length > 1 && int.TryParse(args[1], out int fx) ? fx : 300;
            GameRuntime.EnsureInitialized();
            var rng = new Random(20260704);
            var seen = new Dictionary<string, int>();
            int ok = 0, failed = 0;
            for (int i = 0; i < n; i++)
            {
                string ctx = "?";
                try
                {
                    (GameHost host, CombatScenario.Spec spec) =
                        CombatScenario.Create($"F{i}", rng, characterName: null, elitePct: 0.25, bossPct: 0.10);
                    ctx = $"{spec.Character}/{spec.Encounter}/{spec.RoomType}";
                    PlayFightToEnd(host, rng);
                    ok++;
                }
                catch (Exception ex)
                {
                    failed++;
                    string key = ex.Message.Split('\n')[0].Trim();
                    if (!seen.ContainsKey(key))
                    {
                        seen[key] = 0;
                        Console.Error.WriteLine($"\n===== FIRST OCCURRENCE: {key}\n  ctx (char/encounter/room) = {ctx}\n{ex}\n");
                    }
                    seen[key] = seen[key] + 1;
                }
                if ((i + 1) % 50 == 0)
                {
                    Console.Error.WriteLine($"[fuzz] {i + 1}/{n}  ok={ok} failed={failed}");
                }
            }
            Console.Error.WriteLine($"\n[fuzz] DONE {n} fights: ok={ok} failed={failed}");
            foreach (KeyValuePair<string, int> kv in seen.OrderByDescending(k => k.Value))
            {
                Console.Error.WriteLine($"  {kv.Value,4}x  {kv.Key}");
            }
            return failed == 0 ? 0 : 1;
        }

        try
        {
            var server = new TrainingEnvironmentServer();
            server.Serve(new StreamLineChannel(Console.In, Console.Out));
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[agent-host] fatal: {ex}");
            return 1;
        }
    }

    /// <summary>Play a scenario fight to combat end with a random legal policy — the same "is there still
    /// a combat move?" stop condition the trainer uses (see BuildScenarioObservation). Mirrors the Python
    /// rollout's advance+step so it hits the same code paths (killing-blow reward flow, mid-combat choices,
    /// enemy-turn resolution) that throw in random rollouts.</summary>
    private static void PlayFightToEnd(GameHost host, Random rng)
    {
        for (int guard = 0; guard < 5000; guard++)
        {
            GameState state = host.GetState();
            if (state.IsGameOver)
            {
                return;
            }
            System.Collections.Generic.IReadOnlyList<GameOption> opts = host.ListOptions();
            bool canFight = host.InCombat && opts.Any(o =>
                o.Kind is OptionKind.PlayCard or OptionKind.EndTurn
                    or OptionKind.UsePotion or OptionKind.DiscardPotion or OptionKind.SelectCards);
            if (!canFight)
            {
                return;   // combat over (won or lost) — the trainer resets here
            }
            host.Apply(opts[rng.Next(opts.Count)]);
        }
    }
}
