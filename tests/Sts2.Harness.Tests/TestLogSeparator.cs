using System;
using System.Reflection;
using Xunit.Sdk;

// Apply to every test in the assembly: the game logs gameplay (card plays, monster moves,
// rewards) to stdout, and tests run sequentially into one continuous stream. Printing a banner
// with the test name before/after each test makes it clear which output belongs to which test.
[assembly: Sts2.Harness.Tests.TestLogSeparator]

namespace Sts2.Harness.Tests;

/// <summary>
/// Writes a banner with the test's name to stdout around each test, so the game's interleaved
/// gameplay logging is attributable to a specific test. Writes to <see cref="Console.Out"/> —
/// the same sink the shim's <c>GD.Print</c> uses — so the banner orders correctly with the logs.
/// </summary>
[AttributeUsage(AttributeTargets.Assembly | AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
public sealed class TestLogSeparatorAttribute : BeforeAfterTestAttribute
{
    public override void Before(MethodInfo methodUnderTest) =>
        Console.Out.WriteLine($"{Environment.NewLine}========== TEST: {Name(methodUnderTest)} ==========");

    public override void After(MethodInfo methodUnderTest) =>
        Console.Out.WriteLine($"========== END:  {Name(methodUnderTest)} =========={Environment.NewLine}");

    private static string Name(MethodInfo m) => $"{m.DeclaringType?.Name}.{m.Name}";
}
