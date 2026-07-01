#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Extract the English localization tables from Slay the Spire 2's .pck into lib/localization/,
  so the Lts2.Localization library can feed real names/descriptions to the game's loc system.

.DESCRIPTION
  The loc tables (res://localization/eng/*.json — plain JSON key→text dicts) live only inside the
  ~1.9 GB game .pck, which is gitignored copyrighted content. This script uses GDRE Tools
  (gdsdecomp) to extract just that subset. It is idempotent: it skips when the tables are already
  present unless -Force is given. Run it once per fresh clone (the output lands under the
  already-gitignored lib/).

.PARAMETER Pck
  Path to SlayTheSpire2.pck. Defaults to the Steam install, then the STS2_PCK env var.

.PARAMETER Gdre
  Path to gdre_tools.exe. Defaults to the winget install location, then PATH, then GDRE_TOOLS env var.

.PARAMETER Force
  Re-extract even if the tables already exist.

.EXAMPLE
  pwsh scripts/extract-localization.ps1
#>
[CmdletBinding()]
param(
    [string]$Pck,
    [string]$Gdre,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outRoot  = Join-Path $repoRoot 'lib'                       # gdre writes <out>/localization/...
$locDir   = Join-Path $outRoot 'localization'
$engProbe = Join-Path $locDir 'eng/cards.json'

function Resolve-Pck {
    param([string]$explicit)
    $candidates = @(
        $explicit,
        $env:STS2_PCK,
        'D:\Steam\steamapps\common\Slay the Spire 2\SlayTheSpire2.pck',
        'C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\SlayTheSpire2.pck'
    ) | Where-Object { $_ }
    foreach ($c in $candidates) { if (Test-Path $c) { return (Resolve-Path $c).Path } }
    throw "Could not find SlayTheSpire2.pck. Pass -Pck <path> or set `$env:STS2_PCK."
}

function Resolve-Gdre {
    param([string]$explicit)
    if ($explicit -and (Test-Path $explicit)) { return (Resolve-Path $explicit).Path }
    if ($env:GDRE_TOOLS -and (Test-Path $env:GDRE_TOOLS)) { return (Resolve-Path $env:GDRE_TOOLS).Path }
    $winget = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path $winget) {
        $found = Get-ChildItem -Path $winget -Recurse -Filter 'gdre_tools.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    $cmd = Get-Command 'gdre_tools.exe' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Could not find gdre_tools.exe. Pass -Gdre <path> or set `$env:GDRE_TOOLS. Install with: winget install GDRETools.gdsdecomp"
}

if ((Test-Path $engProbe) -and -not $Force) {
    $count = (Get-ChildItem (Join-Path $locDir 'eng') -Filter '*.json').Count
    Write-Host "Localization already extracted ($count tables in $locDir). Use -Force to re-extract." -ForegroundColor Green
    exit 0
}

$pckPath  = Resolve-Pck  $Pck
$gdrePath = Resolve-Gdre $Gdre
Write-Host "PCK : $pckPath"
Write-Host "GDRE: $gdrePath"

if (Test-Path $locDir) { Remove-Item -Recurse -Force $locDir }
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

# Extract only the English tables + completion file, recreating the res:// tree under lib/.
& $gdrePath --headless `
    --extract="$pckPath" `
    --output-dir="$outRoot" `
    --include="res://localization/eng/*" `
    --include="res://localization/completion.json" | Out-Null

if (-not (Test-Path $engProbe)) {
    throw "Extraction did not produce $engProbe — check the gdre output above."
}

$count = (Get-ChildItem (Join-Path $locDir 'eng') -Filter '*.json').Count
Write-Host "Extracted $count localization tables to $locDir" -ForegroundColor Green
