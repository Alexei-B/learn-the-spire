using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

public sealed class BootstrapSmokeTests
{
    private readonly ITestOutputHelper _out;

    public BootstrapSmokeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void StartNewRun_CreatesRunWithOnePlayer()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");

        Assert.NotNull(host.Run);
        Player player = host.Run.Players.Single();
        _out.WriteLine($"Seed={host.Seed} character={player.Character.GetType().Name} hp={player.Creature.CurrentHp}/{player.Creature.MaxHp} gold={player.Gold}");

        Assert.True(player.Creature.MaxHp > 0);
        Assert.True(player.Deck.Cards.Count > 0);
    }
}
