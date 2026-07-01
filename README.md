
# Learn the Spire

This is a project I've been working on for fun, to test out machine learning ideas on Slay the Spire 2.

## What does this do?

Right now, the project consists of a few .NET libraries and applications:

1. `Lts2.Harness` - A headless library for running Slay the Spire 2
2. `Lts2.GodotShim` - A library that enables `Lts2.Harness` to run free of it's dependencies
3. `Lts2.Localization` - A library for reading localization information from Sts2 data files
4. `Lts2.Tui` - A terminal UI for testing the headless STS 2 harness

Basically right now the project enables you to run STS 2 games without booting the real game. You still of course need
to own a copy of the game and have all the game files downloaded.

## Getting started (Windows)

This assumes you're on Windows, you own Slay the Spire 2, and it's installed via Steam. None of the game's
binaries or assets are in this repo (they're copyrighted), so the one manual step is copying a handful of the
game's own DLLs into a local, gitignored `lib/` folder. From scratch:

### 1. Prerequisites

- **[.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0)** — version `9.0.315` or newer
  (`global.json` pins it, rolling forward to the latest 9.0.3xx you have installed). Check with `dotnet --version`.
- **Git**, and **PowerShell 7** (`pwsh`) if you want localized text (step 4).
- A legitimate **Slay the Spire 2** install. The files this project needs live in the game's
  `data_sts2_windows_x86_64` folder — typically
  `C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\data_sts2_windows_x86_64\`

### 2. Clone

```powershell
git clone https://github.com/Alexei-B/learn-the-spire
cd learn-the-spire
```

### 3. Copy the game binaries into `lib/`

`Lts2.Harness` references the real `sts2.dll` and the (non-Godot) libraries it depends on out of a
gitignored `lib/` folder. Copy every managed DLL from the game's data folder **except `GodotSharp.dll`** —
the project ships its own drop-in replacement for that (the `Lts2.GodotShim` project builds an assembly
literally named `GodotSharp`), and having the real one in `lib/` would conflict with it.

```powershell
# Point this at YOUR install's data folder:
$game = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\data_sts2_windows_x86_64"

New-Item -ItemType Directory -Force lib | Out-Null
Copy-Item "$game\*.dll" lib\ -Exclude "GodotSharp*.dll"
Copy-Item "$game\sts2.xml" lib\ -ErrorAction SilentlyContinue   # optional: XML docs for reference
```

That should leave you with `sts2.dll`, `0Harmony.dll`, `MonoMod.*`, `SmartFormat*`, `System.IO.Hashing.dll`,
`JetBrains.Annotations.dll`, `Steamworks.NET.dll`, `Sentry.dll`, the `SharpGen.*`/`Vortice.*` DLLs, and so on —
but **not** `GodotSharp.dll`. (If you accidentally copy it in, just delete `lib\GodotSharp.dll` and rebuild.)

### 4. (Optional but recommended) Extract localization

Without this, everything still works — the UI just shows model ids (e.g. `BASH`) instead of real names and
descriptions. To get the real text, extract the English localization tables from the game's `.pck` with
[GDRE Tools](https://github.com/GDRETools/gdsdecomp):

```powershell
winget install GDRETools.gdsdecomp    # one-time install of the extractor
pwsh scripts/extract-localization.ps1 # writes lib/localization/
```

The script auto-detects the Steam `.pck` and the winget-installed `gdre_tools.exe`; if yours are elsewhere,
pass `-Pck <path-to-SlayTheSpire2.pck>` / `-Gdre <path-to-gdre_tools.exe>`, or set the `STS2_PCK` /
`GDRE_TOOLS` environment variables. It's idempotent — re-run with `-Force` to refresh.

### 5. Build, test, run

```powershell
dotnet build learn-the-spire.sln     # build everything
dotnet test  learn-the-spire.sln     # run the (seeded, deterministic) test suite
dotnet run --project src/Lts2.Tui    # play a run in the terminal UI
```

The TUI opens a New Run dialog (character / ascension / seed), then plays a full single-player run through
the harness. See [`src/Lts2.Tui/README.md`](./src/Lts2.Tui/README.md) for the controls and layout.

### Optional: `refsrc/` for development

You don't need this to build or run — only if you want to read the game's internals. `refsrc/` holds a full
ILSpy decompile of `sts2.dll` (and the real `GodotSharp.dll`) for reference; it's gitignored and never
compiled. Grep it when you want to understand how a piece of the game works.

## AI Disclosure

This project is currently 100% vide coded - I do this just for fun and to learn, so that suits my needs.
If that bothers you; fair enough.

I've been using Claude for this project, mostly Opus 4.8, but also Sonnet.

## License

[Slay the Spire 2](https://store.steampowered.com/app/2868840/Slay_the_Spire_2/) is of course
© [Mega Crit Games](https://www.megacrit.com/), who own the game and all it's IP.
None of the game code or assets are distributed within this repository.
This project isn't affiliated with Mega Crit Games in any way.

[Godot](https://godotengine.org/) is supported by the [Godot Foundation](https://godot.foundation/).
This repo contains a shim with an API written to match the API discovered by decompiling Godot code. for technical
reasons shares the same binary name as the library `GodotSharp` - this is purely functional, it is not intended to be
used in place of the real library, nor should it be confused for that real library.
For more information, see the [Godot License](https://github.com/godotengine/godot/blob/master/LICENSE.txt).

This is distributed under the unlicense. See [LICENSE](./LICENSE) for details, but basically just do whatever you want.

## Contributing

If you would like to fork this repo or just copy code from it; be my guest. I'm not planning on maintaining this
project since it's just for my own enjoyment.

## Spire Codex

I've used similar techniques to the [spire-codex](https://spire-codex.com/) project to decompile game code and unpack
game assets. This was extremely helpful, so thank you to the people behind spire-codex.
