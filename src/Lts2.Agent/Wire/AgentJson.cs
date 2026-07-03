using System.Text.Json;
using System.Text.Json.Serialization;

namespace Lts2.Agent.Wire;

/// <summary>
/// The single <see cref="JsonSerializerOptions"/> for the agent wire protocol, shared by every
/// message on both sides of the boundary. Enums serialize by their name (so a Python agent sees
/// <c>"Combat"</c>/<c>"PlayCard"</c>, not opaque ints), property names are camelCase, and nulls are
/// omitted to keep the observation compact. Keeping one instance here guarantees the training
/// environment and the evaluation decision-server agree byte-for-byte on the schema.
/// </summary>
public static class AgentJson
{
    /// <summary>The current wire protocol version (bumped on any breaking schema change).</summary>
    public const int ProtocolVersion = 1;

    /// <summary>The shared serializer options. Immutable after construction; safe to reuse.</summary>
    public static readonly JsonSerializerOptions Options = Build();

    private static JsonSerializerOptions Build()
    {
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            // Observations are one-per-line; never emit embedded newlines.
            WriteIndented = false,
        };
        options.Converters.Add(new JsonStringEnumConverter());
        return options;
    }
}
