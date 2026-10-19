# Shared configuration. Dot-source it: . "$PSScriptRoot\config.ps1"
# All heavy files (weights, ACE-Step venv, caches) live in MODELS_DIR.
# MODELS_DIR is an ENVIRONMENT VARIABLE of the machine: to keep heavy files on another drive set it once, e.g.
#   setx MODELS_DIR D:\Models        (then open a new terminal)
# When it is not set, the default <project>\models is used (self-contained, git-ignored).

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ModelsDirIsSet = [bool]$env:MODELS_DIR
if (-not $ModelsDirIsSet) { $env:MODELS_DIR = Join-Path $ProjectRoot 'models' }
$ModelsDir = $env:MODELS_DIR
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# Ukrainian/emoji output from the python scripts
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = '1'

# Main pipeline (Meta AudioCraft)
$env:HF_HOME = $ModelsDir
$env:AUDIOCRAFT_CACHE_DIR = Join-Path $ModelsDir 'audiocraft'
$env:TORCH_HOME = Join-Path $ModelsDir 'torch'
$MainPython = Join-Path $ProjectRoot 'env\Scripts\python.exe'

# ACE-Step 1.5 (code is cloned into the project, everything heavy goes to MODELS_DIR)
$AceStepDir = Join-Path $ProjectRoot 'ace-step'
$env:ACESTEP_DIR = $AceStepDir
$env:ACESTEP_CHECKPOINTS_DIR = Join-Path $ModelsDir 'ace-step\checkpoints'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ModelsDir 'ace-step\venv'
$env:UV_CACHE_DIR = Join-Path $ModelsDir 'uv-cache'
# models drive differs from the project drive: hardlinks are impossible, copy silently
$env:UV_LINK_MODE = 'copy'
$AceStepPython = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'

# Run a native command and fail loudly on a non-zero exit code.
function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}
