# Route B: play the song again with the bank's real samples (drums replaced, power-chord guitars added).
# Usage: native_arrange.ps1 <song-root> [--tracks a,b] [--start S --length L] [--seed N] [--tag name] [--vocals 1] [--bass 1]
#        [--keep-other 0] [--guitar 1] [--drums 1] [--no-crash]       (main env; tools\native_arrange.py)
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $MainPython)) { throw 'Main environment is missing: run.bat setup-all --main-only (or setup-all)' }
if (-not $Extra) { throw 'Usage: run.bat native-arrange <song-root> [--tracks a,b] [--start S --length L] [--seed N]' }
Invoke-Checked $MainPython (@('-u', (Join-Path $ProjectRoot 'tools\native_arrange.py')) + $Extra)
