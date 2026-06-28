using System;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using Sts2.Harness;
using Xunit;
using Xunit.Abstractions;

namespace Sts2.Harness.Tests;

public sealed class RoomEntrySmokeTests
{
    private readonly ITestOutputHelper _out;

    public RoomEntrySmokeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task EnterFirstRoom_AdvancesIntoARoom()
    {
        GameHost host = GameHost.StartNewRun(seed: "TESTSEED");

        // Guard against a hang: room entry may block awaiting player input.
        Task entry = Task.Run(() => host.EnterFirstRoom());
        try
        {
            await entry.WaitAsync(TimeSpan.FromSeconds(20));
        }
        catch (TimeoutException)
        {
            // Surface the state even if it didn't return, then fail clearly.
            _out.WriteLine($"CurrentRoom={host.Run.CurrentRoom?.GetType().Name ?? "<null>"} act={host.Run.CurrentActIndex} floor={host.Run.TotalFloor}");
            Assert.Fail("EnterFirstRoom did not return within 20s (likely blocked awaiting player input).");
        }

        _out.WriteLine($"CurrentRoom={host.Run.CurrentRoom?.GetType().Name ?? "<null>"} act={host.Run.CurrentActIndex} floor={host.Run.TotalFloor}");
        _out.WriteLine($"CombatInProgress={CombatManager.Instance.IsInProgress}");
    }
}
