# Route B: slice the stems of reference tracks into a bank of real samples (drum hits, guitar chords).
# Usage: native_bank.ps1 <reference-dir | song-root\reference> [...] [--redo] [--bank <dir>]     (main env; tools\native_bank.py)
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $MainPython)) { throw 'Main environment is missing: run.bat setup-all --main-only (or setup-all)' }
if (-not $Extra) { throw 'Usage: run.bat native-bank <song-root>\reference [--redo]' }
Invoke-Checked $MainPython (@('-u', (Join-Path $ProjectRoot 'tools\native_bank.py')) + $Extra)
