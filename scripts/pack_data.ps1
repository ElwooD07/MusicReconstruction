# Copies the data that git does not carry to a transfer folder (USB drive, network share): the song folder
# (with orig\ stems), the reference tracks (with their stems) and, optionally, the ACE-Step weights.
# The project itself goes through git (clone it on the new machine); env\, ace-step\, models\, temp\ are not copied.
# Usage: pack_data.ps1 --dest <folder> [--song <folder>] [--tracks <folder>] [--weights]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

$Extra = @($Extra)
function Option($name) {
    $i = [array]::IndexOf($Extra, $name)
    if ($i -ge 0 -and $Extra.Count -gt $i + 1) { return $Extra[$i + 1].Trim('"') }
}
$Dest = Option '--dest'
if (-not $Dest) { throw 'Usage: pack-data --dest <folder> [--song <folder>] [--tracks <folder>] [--weights]' }
$Song = Option '--song'
$Tracks = Option '--tracks'

function Copy-Folder($src, $dst) {
    if (-not (Test-Path $src -PathType Container)) { throw "Folder not found: $src" }
    Write-Host "copy $src -> $dst"
    # /E subfolders, /XO skip files that are already there and not older; gen\ holds only experiments
    & robocopy $src $dst /E /XO /XD gen /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
    $global:LASTEXITCODE = 0   # robocopy's 1-7 mean "copied", not an error
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
if ($Song) { Copy-Folder $Song (Join-Path $Dest 'song') }
if ($Tracks) { Copy-Folder $Tracks (Join-Path $Dest 'tracks') }
if ($Extra -contains '--weights') { Copy-Folder $env:ACESTEP_CHECKPOINTS_DIR (Join-Path $Dest 'ace-step\checkpoints') }

Write-Host @"

Done: $Dest
On the new machine: clone the repository, then run.bat setup-all --check and run.bat setup-all.
To reuse the weights put '$Dest\ace-step\checkpoints' into <MODELS_DIR>\ace-step\checkpoints before setup-all.
If the stems of the tracks are in tracks\reference, copy that folder into the song folder (next to orig\).
"@
