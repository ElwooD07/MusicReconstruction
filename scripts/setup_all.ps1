# One-step preparation of a NEW machine for the UI workflow (also a health check).
# Usage: setup_all.ps1 [--check] [--full] [--models acestep-v15-base,acestep-v15-turbo]
#   --full    also install the MusicGen stack (audiocraft, xformers, spacy...) via setup_main.ps1; only reconstruct.py
#             needs it, the UI does not
#   --check   only report what is installed / missing, change nothing
#   --main-only  only the main environment + demucs weights (~8 GB): enough for check-tracks / stems, no ACE-Step
#   --models  ACE-Step DiT checkpoints to download (default: base for Lego + turbo for Cover)
# Needs: Python 3.10-3.12 and git in PATH, an NVIDIA driver, internet. Everything heavy goes to MODELS_DIR.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

$Extra = @($Extra)
$checkOnly = $Extra -contains '--check'
$full = $Extra -contains '--full'
$mainOnly = $Extra -contains '--main-only'
$models = @('acestep-v15-base', 'acestep-v15-turbo')
$i = [array]::IndexOf($Extra, '--models')
if ($i -ge 0 -and $Extra.Count -gt $i + 1) { $models = $Extra[$i + 1] -split ',' }

# keep pip's cache (GBs of wheels) off the system drive too
$env:PIP_CACHE_DIR = Join-Path $ModelsDir 'pip-cache'
$envScripts = Join-Path $ProjectRoot 'env\Scripts'
if (Test-Path $envScripts) { $env:PATH = "$envScripts;$env:PATH" }   # uv installed into ./env is found too

function Row($name, $ok, $detail) {
    Write-Host ('{0} {1,-24} {2}' -f $(if ($ok) { '[ok]' } else { '[--]' }), $name, $detail)
}

function Show-Status {
    # Windows PowerShell 5.1 turns redirected stderr (2>$null / 2>&1) into terminating errors under 'Stop'; a Python
    # warning must not abort the report
    $ErrorActionPreference = 'Continue'
    Write-Host "`n--- Status (models dir: $ModelsDir) ---"
    Row 'MODELS_DIR (env variable)' $ModelsDirIsSet $(if ($ModelsDirIsSet) { $ModelsDir } else {
        "not set: using the default <project>\models. To keep weights and caches on another drive: setx MODELS_DIR <path>, then open a new terminal" })
    $py = Get-Command python -ErrorAction SilentlyContinue
    $pyVer = if ($py) { (& python --version 2>&1) -replace 'Python ', '' } else { '' }
    Row 'Python' ($pyVer -match '^3\.(10|11|12)\.') "$pyVer (3.10-3.12 needed; 3.11 was used)"
    Row 'git' ([bool](Get-Command git -ErrorAction SilentlyContinue)) ''
    $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    $vram = if ($gpu) { [int](($gpu -split ',')[1] -replace '[^0-9]', '') } else { 0 }
    Row 'NVIDIA GPU' ($vram -gt 0) "$gpu"
    $drvText = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    $drv = 0.0
    [void][double]::TryParse(($drvText -replace '[^0-9.]', ''), [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$drv)
    Row 'NVIDIA driver' ($drv -ge 570) "$drvText (the ACE-Step environment uses CUDA 12.8 wheels; NVIDIA documents driver 570+ for them)"
    if ($vram -gt 0 -and $vram -lt 12000) {
        Write-Host '     note: under 12 GB VRAM. ACE-Step recommends the turbo model for 6-8 GB (README, VRAM table); Lego needs base,'
        Write-Host '           which is NOT verified on 8 GB: try short fragments (20-30 s) and close other GPU programs.'
    }
    $drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($ModelsDir).Substring(0, 1)) -ErrorAction SilentlyContinue
    if ($drive) { Row 'Free disk (models)' ($drive.Free -gt 40GB) ('{0:N0} GB free; about 35 GB needed' -f ($drive.Free / 1GB)) }
    $mainOk = $false
    if (Test-Path $MainPython) {
        $r = & $MainPython -c "import torch, demucs, av, librosa, soundfile; print(torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | Select-Object -Last 1
        $mainOk = $r -match 'cuda True'
        Row 'Main env (demucs...)' $mainOk "$r"
    } else { Row 'Main env (demucs...)' $false 'not created' }
    if (Test-Path $MainPython) {
        $ac = & $MainPython -c "import audiocraft; print(audiocraft.__version__)" 2>$null | Select-Object -Last 1
        Row 'audiocraft (MusicGen)' ([bool]$ac) $(if ($ac) { "$ac" } else { 'not installed; needed only by reconstruct.py (setup-all --full)' })
    }
    Row 'uv' ([bool](Get-Command uv -ErrorAction SilentlyContinue)) '(installs the ACE-Step environment)'
    Row 'ACE-Step code' (Test-Path (Join-Path $AceStepDir 'pyproject.toml')) $AceStepDir
    Row 'ACE-Step environment' (Test-Path $AceStepPython) $env:UV_PROJECT_ENVIRONMENT
    $want = (Get-Content (Join-Path $ProjectRoot 'ace-step.version') -First 1 -ErrorAction SilentlyContinue)
    $have = if (Test-Path (Join-Path $AceStepDir '.git')) { (& git -C $AceStepDir rev-parse HEAD 2>$null) } else { '' }
    Row 'ACE-Step version' ($want -and $have -eq $want.Trim()) $(if ($have) { "$have (pinned in ace-step.version)" } else { 'not installed' })
    foreach ($m in $models + 'vae') {
        Row "weights: $m" (Test-Path (Join-Path $env:ACESTEP_CHECKPOINTS_DIR $m)) ''
    }
}

if ($checkOnly) { Show-Status; return }

Write-Host '=== 1/5 main environment (torch + demucs) ==='
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python was not found in PATH (install 3.11 from python.org)' }
if (-not (Test-Path $MainPython)) { Invoke-Checked python @('-m', 'venv', (Join-Path $ProjectRoot 'env')) }
$envScripts = Join-Path $ProjectRoot 'env\Scripts'
$env:PATH = "$envScripts;$env:PATH"
Invoke-Checked $MainPython @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet')
Invoke-Checked $MainPython @('-m', 'pip', 'install', 'torch==2.14.1', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu126')
Invoke-Checked $MainPython @('-m', 'pip', 'install', '-r', (Join-Path $ProjectRoot 'requirements-main.txt'))

if ($full) {
    Write-Host '=== 1b: MusicGen stack (audiocraft, xformers, spacy...) ==='
    & "$PSScriptRoot\setup_main.ps1"
}

function Install-DemucsWeights {
    Invoke-Checked $MainPython @('-c', "from demucs.pretrained import get_model; get_model('htdemucs_ft'); get_model('htdemucs'); print('demucs ok')")
}
if ($mainOnly) {
    Write-Host '=== demucs weights ==='
    Install-DemucsWeights
    Show-Status
    Write-Host "`nMain environment ready: run.bat check-tracks <folder> --demux. For the rest: run.bat setup-all"
    return
}

Write-Host '=== 2/5 uv ==='
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Invoke-Checked $MainPython @('-m', 'pip', 'install', 'uv') }

Write-Host '=== 3/5 ACE-Step code + environment ==='
if (-not (Test-Path $AceStepPython)) { & "$PSScriptRoot\setup_acestep.ps1" }

Write-Host "=== 4/5 ACE-Step weights: vae/LM + $($models -join ', ') ==="
$code = @"
import sys
sys.path.insert(0, r'$AceStepDir')
from acestep.model_downloader import ensure_main_model, ensure_dit_model, get_checkpoints_dir
c = get_checkpoints_dir()
print('main', ensure_main_model(checkpoints_dir=c))
for m in '$($models -join ',')'.split(','):
    print(m, ensure_dit_model(m, checkpoints_dir=c))
"@
Invoke-Checked $AceStepPython @('-c', $code)

Write-Host '=== 5/5 demucs weights ==='
Install-DemucsWeights

Show-Status
Write-Host "`nReady. Start the UI with: run.bat ui"
