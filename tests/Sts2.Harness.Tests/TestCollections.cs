// The game stores all run/combat/save state in process-wide singletons, so tests
// that drive a run cannot run in parallel — they would corrupt each other's shared
// state. Disable parallelization across the whole assembly.
[assembly: Xunit.CollectionBehavior(DisableTestParallelization = true)]
