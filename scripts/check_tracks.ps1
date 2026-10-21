# Checks a pack of reference tracks against the PLAN.md requirements; --demux also splits them into stems.
# Usage: check_tracks.ps1 <folder> [--demux] [--out <dir>]     (main env; see tools\check_tracks.py)
param(
    [string]$Folder,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $MainPython)) { throw 'Main environment is missing: run.bat setup-all --main-only (or setup-all)' }
if (-not $Folder) { $Folder = Read-Host 'Folder with the reference tracks' }
$pyArgs = @('-u', (Join-Path $ProjectRoot 'tools\check_tracks.py'), $Folder.Trim('"'))
if ($Extra) { $pyArgs += $Extra }
Invoke-Checked $MainPython $pyArgs
