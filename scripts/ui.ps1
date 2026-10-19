# Tuning UI (Gradio) inside the ACE-Step venv.
# Usage: ui.ps1 [--port 7860] [--no-browser]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $AceStepPython)) { & "$PSScriptRoot\setup_acestep.ps1" }
# Gradio copies every audio file it serves into a temp folder: keep that off the nearly full C: drive
$env:GRADIO_TEMP_DIR = Join-Path $ModelsDir 'gradio-tmp'
$pyArgs = @('-u', (Join-Path $ProjectRoot 'ui.py'))
if ($Extra) { $pyArgs += $Extra }
Invoke-Checked $AceStepPython $pyArgs
