# Transforms the music in a folder using reconstruct.py (vocals are not touched).
# Usage: reconstruct.ps1 [folder] [prompt]
param(
    [string]$Folder,
    [string]$Prompt
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $MainPython)) { & "$PSScriptRoot\setup_main.ps1" }

Write-Host '================================================================='
Write-Host 'Transforming music using GPU'
Write-Host '================================================================='

if (-not $Folder) { $Folder = Read-Host '1. Drop down the folder with MP3-music you want to transform' }
if (-not $Prompt) { $Prompt = Read-Host '2. Type a prompt for the desired new style of the music (vocals will not be touched)' }
$Folder = $Folder.Trim('"')

Write-Host 'Reconstructing music...'
$pyArgs = @((Join-Path $ProjectRoot 'reconstruct.py'), $Folder)
if ($Prompt) { $pyArgs += $Prompt }
Invoke-Checked $MainPython $pyArgs
