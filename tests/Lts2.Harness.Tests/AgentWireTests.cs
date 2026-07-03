using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Lts2.Agent;
using Lts2.Agent.Wire;
using Lts2.Harness;
using Xunit;

namespace Lts2.Harness.Tests;

/// <summary>
/// Tests for the cross-process agent seam (<c>src/Lts2.Agent</c>): the wire serialization of an
/// observation, the <see cref="ProcessDecisionEngine"/> client's index-mapping and graceful decline,
/// and the <see cref="TrainingEnvironmentServer"/> reset/step/close loop over an in-memory channel.
/// All deterministic from a fixed seed; the real subprocess/stdio transport is exercised via the
/// Python round-trip, while these drive the same logic through <see cref="StreamLineChannel"/> over
/// string streams.
/// </summary>
public sealed class AgentWireTests
{
    [Fact]
    public void Observation_SerializesState_WithStringEnums_AndOptionIndicesStable()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();

        Observation obs = Observation.From(state, options);
        string json = JsonSerializer.Serialize(obs, AgentJson.Options);

        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;

        // Enums serialize by name, not as ints.
        Assert.Equal("Combat", root.GetProperty("state").GetProperty("phase").GetString());
        Assert.False(root.GetProperty("done").GetBoolean());
        Assert.Equal(AgentJson.ProtocolVersion, root.GetProperty("protocolVersion").GetInt32());

        // The options list round-trips 1:1 with ListOptions, so an action index is meaningful.
        JsonElement wireOptions = root.GetProperty("options");
        Assert.Equal(options.Count, wireOptions.GetArrayLength());
        Assert.All(
            wireOptions.EnumerateArray(),
            o => Assert.False(string.IsNullOrEmpty(o.GetProperty("kind").GetString())));

        // The info block carries the reward scalars.
        JsonElement info = root.GetProperty("info");
        Assert.Equal(state.Score, info.GetProperty("score").GetInt32());
        Assert.Equal(state.Players[0].CurrentHp, info.GetProperty("players")[0].GetProperty("currentHp").GetInt32());
    }

    [Fact]
    public void ProcessDecisionEngine_MapsScoresBackToOptionsByIndex()
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();

        var channel = new FakeChannel
        {
            Response = """{"scores":[{"index":0,"score":9.5,"rationale":"pick first"}]}""",
        };
        var engine = new ProcessDecisionEngine("Fake", channel);

        IReadOnlyList<ScoredOption> scored = engine.Evaluate(state, options);

        ScoredOption only = Assert.Single(scored);
        Assert.Same(options[0], only.Option);
        Assert.Equal(9.5, only.Score);
        Assert.Equal("pick first", only.Rationale);
        Assert.Same(options[0], engine.Recommend(state, options));

        // The request it sent is a well-formed evaluate message carrying the same option count.
        using JsonDocument req = JsonDocument.Parse(channel.LastRequest!);
        Assert.Equal("evaluate", req.RootElement.GetProperty("type").GetString());
        Assert.Equal(options.Count, req.RootElement.GetProperty("options").GetArrayLength());
    }

    [Theory]
    [InlineData(null)]                              // unavailable / dead process
    [InlineData("not json at all")]                // malformed reply
    [InlineData("""{"scores":[{"index":9999}]}""")] // out-of-range index dropped
    [InlineData("""{"scores":[]}""")]               // explicit decline
    public void ProcessDecisionEngine_DeclinesGracefully_OnBadResponses(string? response)
    {
        GameHost host = TestNav.MoveIntoFirstCombat("TESTSEED");
        GameState state = host.GetState();
        IReadOnlyList<GameOption> options = host.ListOptions();

        var engine = new ProcessDecisionEngine("Fake", new FakeChannel { Response = response });

        Assert.Empty(engine.Evaluate(state, options));
        Assert.Null(engine.Recommend(state, options));
    }

    [Fact]
    public void TrainingEnvironmentServer_ResetStepClose_ProducesObservations()
    {
        string input = string.Join("\n",
            """{"cmd":"reset","seed":"TESTSEED","character":"Iron"}""",
            """{"cmd":"step","index":0}""",
            """{"cmd":"close"}""") + "\n";

        var output = new System.IO.StringWriter();
        var channel = new StreamLineChannel(new System.IO.StringReader(input), output);
        new TrainingEnvironmentServer().Serve(channel);

        string[] lines = output.ToString().Split('\n', System.StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(3, lines.Length);

        // reset → an opening observation with a phase, options, and reward scalars.
        using JsonDocument reset = JsonDocument.Parse(lines[0]);
        Assert.False(reset.RootElement.GetProperty("done").GetBoolean());
        Assert.True(reset.RootElement.GetProperty("options").GetArrayLength() > 0);
        Assert.True(reset.RootElement.GetProperty("info").GetProperty("score").GetInt32() >= 0);

        // step → another observation (the game advanced).
        using JsonDocument step = JsonDocument.Parse(lines[1]);
        Assert.True(step.RootElement.TryGetProperty("state", out _));

        // close → an acknowledgement.
        using JsonDocument close = JsonDocument.Parse(lines[2]);
        Assert.True(close.RootElement.GetProperty("ok").GetBoolean());
    }

    [Fact]
    public void TrainingEnvironmentServer_ReportsErrors_WithoutStopping()
    {
        // An out-of-range step index after a reset errors, but the server keeps serving (close still acks).
        string input = string.Join("\n",
            """{"cmd":"reset","seed":"TESTSEED"}""",
            """{"cmd":"step","index":9999}""",
            """{"cmd":"close"}""") + "\n";

        var output = new System.IO.StringWriter();
        var channel = new StreamLineChannel(new System.IO.StringReader(input), output);
        new TrainingEnvironmentServer().Serve(channel);

        string[] lines = output.ToString().Split('\n', System.StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(3, lines.Length);

        using JsonDocument error = JsonDocument.Parse(lines[1]);
        Assert.True(error.RootElement.TryGetProperty("error", out JsonElement message));
        Assert.Contains("range", message.GetString(), System.StringComparison.OrdinalIgnoreCase);

        using JsonDocument close = JsonDocument.Parse(lines[2]);
        Assert.True(close.RootElement.GetProperty("ok").GetBoolean());
    }

    private sealed class FakeChannel : IDecisionChannel
    {
        public string? Response { get; init; }
        public string? LastRequest { get; private set; }

        public string? SendRequest(string requestLine)
        {
            LastRequest = requestLine;
            return Response;
        }
    }
}
