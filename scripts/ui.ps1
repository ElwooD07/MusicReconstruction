# UI of route B (Gradio) in the main env: song, reference tracks, sample bank, arrangement.
# Usage: ui.ps1 [--port 7860] [--no-browser]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $MainPython)) { throw 'Main environment is missing: run.bat setup-all --main-only (or setup-all)' }
# Gradio copies every audio file it serves into a temp folder: keep that off the nearly full C: drive
$env:GRADIO_TEMP_DIR = Join-Path $ModelsDir 'gradio-tmp'
$pyArgs = @('-u', (Join-Path $ProjectRoot 'ui.py'))
if ($Extra) { $pyArgs += $Extra }
Invoke-Checked $MainPython $pyArgs
