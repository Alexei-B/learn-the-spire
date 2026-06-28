using System;
using System.Linq;
using System.Threading.Tasks;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

/// <summary>
/// Exercises the M1 public API surface: <see cref="GameHost.GetState"/>,
/// <see cref="GameHost.ListOptions(ulong)"/> and <see cref="GameHost.Apply"/>.
/// </summary>
public sealed class PublicApiTests
{
    private readonly ITestOutputHelper _out;

    public PublicApiTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void GetState_OnMap_ProjectsRunAndMap()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

        GameState state = host.GetState();

        _out.WriteLine($"phase={state.Phase} act={state.ActIndex} floor={state.Floor}");
        PlayerState player = Assert.Single(state.Players);
        _out.WriteLine($"player {player.Character} hp={player.CurrentHp}/{player.MaxHp} gold={player.Gold} deck={player.Deck.Count} relics={player.Relics.Count}");

        Assert.Equal(GamePhase.Map, state.Phase);
        Assert.True(player.CurrentHp > 0 && player.CurrentHp == player.MaxHp);
        Assert.NotEmpty(player.Deck);
        Assert.Null(player.CombatState);
        Assert.NotNull(state.Map);
        Assert.NotEmpty(state.Map!.Reachable);
        Assert.NotEmpty(state.Map.Points);
        // No combat off the map.
        Assert.Null(state.Combat);
    }

    [Fact]
    public void ListOptions_OnMap_OffersReachableMoves()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");
        host.EnterFirstRoom();

        var options = host.ListOptions();
        foreach (GameOption o in options)
        {
            _out.WriteLine($"{o.Kind}: {o.Description}");
        }

        Assert.NotEmpty(options);
        Assert.All(options, o => Assert.Equal(OptionKind.MoveTo, o.Kind));
        // Each move option matches a reachable coordinate in the snapshot.
        var reachable = host.GetState().Map!.Reachable.ToHashSet();
        Assert.All(options, o => Assert.Contains(o.Coord!.Value, reachable));
    }

    [Fact]
    public void GetState_InCombat_ProjectsHandEnemiesAndIntents()
    {
        GameHost host = MoveIntoFirstCombat("TESTSEED");
        GameState state = host.GetState();

        Assert.Equal(GamePhase.Combat, state.Phase);
        Assert.NotNull(state.Combat);
        PlayerState player = Assert.Single(state.Players);
        PlayerCombatView pcs = Assert.IsType<PlayerCombatView>(player.CombatState);

        _out.WriteLine($"round={state.Combat!.RoundNumber} side={state.Combat.CurrentSide} energy={pcs.Energy}/{pcs.MaxEnergy} hand={pcs.Hand.Count}");
        foreach (EnemyView e in state.Combat.Enemies)
        {
            string intents = string.Join("+", e.Intents.Select(i => i.Damage is int d ? $"{i.Type}({d}x{i.Hits})" : i.Type.ToString()));
            _out.WriteLine($"enemy {e.MonsterId} hp={e.CurrentHp}/{e.MaxHp} hittable={e.IsHittable} intents=[{intents}]");
        }

        Assert.NotEmpty(pcs.Hand);
        Assert.True(pcs.MaxEnergy > 0);
        Assert.NotEmpty(state.Combat.Enemies);
    }

    [Fact]
    public void Capture_IsADetachedSnapshot()
    {
        GameHost host = MoveIntoFirstCombat("TESTSEED");
        GameState before = host.GetState();
        int enemyHpBefore = before.Combat!.Enemies[0].CurrentHp;

        // Play any available attack via the public API.
        GameOption attack = host.ListOptions().First(o => o.Kind == OptionKind.PlayCard && o.TargetCombatId is not null);
        host.Apply(attack);

        GameState after = host.GetState();
        // The earlier snapshot is unchanged; a fresh capture reflects the damage.
        Assert.Equal(enemyHpBefore, before.Combat.Enemies[0].CurrentHp);
        Assert.True(after.Combat!.Enemies.Sum(e => e.CurrentHp) < before.Combat.Enemies.Sum(e => e.CurrentHp));
    }

    [Fact]
    public async Task ListOptionsApply_DrivesAFullCombatToVictory()
    {
        GameHost host = MoveIntoFirstCombat("TESTSEED");

        var t = Task.Run(() => PlayGreedily(host, maxTurns: 50));
        int turns = await t.WaitAsync(TimeSpan.FromSeconds(60));

        GameState end = host.GetState();
        _out.WriteLine($"after {turns} turns: phase={end.Phase} playerHp={end.Players[0].CurrentHp}");
        Assert.False(host.InCombat, "combat should have ended");
        Assert.True(end.Players[0].CurrentHp > 0, "player should have survived the opening fight");
    }

    /// <summary>Greedily play every playable card each turn, then end turn, via the public API.</summary>
    private int PlayGreedily(GameHost host, int maxTurns)
    {
        int turns = 0;
        while (host.InCombat && turns < maxTurns)
        {
            int guard = 0;
            while (host.InCombat && guard++ < 100)
            {
                GameOption? play = host.ListOptions().FirstOrDefault(o => o.Kind == OptionKind.PlayCard);
                if (play is null)
                {
                    break;
                }
                host.Apply(play);
            }

            if (!host.InCombat)
            {
                break;
            }
            GameOption end = host.ListOptions().First(o => o.Kind == OptionKind.EndTurn);
            host.Apply(end);
            turns++;
        }
        return turns;
    }

    private static GameHost MoveIntoFirstCombat(string seed)
    {
        GameHost host = GameHost.StartNewRun(seed);
        host.EnterFirstRoom();
        // The first reachable map point on the opening seed is a monster room.
        GameOption move = host.ListOptions().First(o => o.Kind == OptionKind.MoveTo);
        host.Apply(move);
        Assert.True(host.InCombat, "expected to land in combat after the first move");
        return host;
    }
}
