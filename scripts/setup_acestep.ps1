# Clones ACE-Step 1.5 into ./ace-step (gitignored) and installs its venv into MODELS_DIR.
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

Write-Host "Models directory: $ModelsDir"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/'
}

# The exact ACE-Step commit this project was developed with (ui.py and tools\ use its internals)
$AceStepCommit = (Get-Content (Join-Path $ProjectRoot 'ace-step.version') -First 1).Trim()
if (-not (Test-Path (Join-Path $AceStepDir 'pyproject.toml'))) {
    Write-Host "Fetching ACE-Step 1.5 commit $AceStepCommit into $AceStepDir..."
    New-Item -ItemType Directory -Force -Path $AceStepDir | Out-Null
    Invoke-Checked git @('-C', $AceStepDir, 'init', '--quiet')
    Invoke-Checked git @('-C', $AceStepDir, 'remote', 'add', 'origin', 'https://github.com/ACE-Step/ACE-Step-1.5.git')
    Invoke-Checked git @('-C', $AceStepDir, 'fetch', '--depth', '1', 'origin', $AceStepCommit)
    Invoke-Checked git @('-C', $AceStepDir, 'checkout', '--quiet', 'FETCH_HEAD')
}

Write-Host "Installing dependencies into $env:UV_PROJECT_ENVIRONMENT ..."
Push-Location $AceStepDir
try { Invoke-Checked uv @('sync') } finally { Pop-Location }

Write-Host "Done. Model checkpoints will be downloaded to $env:ACESTEP_CHECKPOINTS_DIR on first use."
