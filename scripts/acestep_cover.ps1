# Runs tools/acestep_cover.py inside the ACE-Step venv.
# Usage: acestep_cover.ps1 [folder] [--strengths 0.5,0.7,1.0]
param(
    [string]$Folder,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $AceStepPython)) { & "$PSScriptRoot\setup_acestep.ps1" }
if (-not $Folder) { $Folder = Read-Host 'Folder with the MP3 and its separated\htdemucs stems' }
$Folder = $Folder.Trim('"')

# Output folder: $env:EXPERIMENT_OUT_DIR (relative to the project) or temp\ace
$outDir = if ($env:EXPERIMENT_OUT_DIR) { $env:EXPERIMENT_OUT_DIR } else { 'temp\ace' }
if (-not [System.IO.Path]::IsPathRooted($outDir)) { $outDir = Join-Path $ProjectRoot $outDir }
$pyArgs = @('-u', (Join-Path $ProjectRoot 'tools\acestep_cover.py'), $Folder, $outDir)
if ($Extra) { $pyArgs += $Extra }
Invoke-Checked $AceStepPython $pyArgs
