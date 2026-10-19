# Creates ./env and installs the packages for reconstruct.py (idempotent).
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

Write-Host '================================================================='
Write-Host 'Transform music using prompt: prepare the environment'
Write-Host '================================================================='

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python was not found in PATH' }

if (-not (Test-Path $MainPython)) {
    Write-Host 'Creating local environment...'
    Invoke-Checked python @('-m', 'venv', (Join-Path $ProjectRoot 'env'))
}
Write-Host "The directory for huggingface is set to $env:HF_HOME"

function Pip { Invoke-Checked $MainPython (@('-m', 'pip') + $args) }

Write-Host 'Installing components...'
Pip install --upgrade pip --quiet

Write-Host '  -[1/4] Installing pytorch for CUDA 12.6...'
Pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Write-Host '  -[2/4] Installing av...'
Pip install av --only-binary=:all: --upgrade --quiet

Write-Host '  -[3/4] Installing Meta AudioCraft...'
Pip install audiocraft --no-deps --quiet
Pip install omegaconf hydra-core einops local-attention treetable submitit sentencepiece encodec flashy num2words spacy gradio torchmetrics hydra-colorlog librosa scipy --quiet
Pip install av julius transformers demucs xformers

Write-Host '  -[4/4] Clearing pip installation cache...'
Pip cache purge --quiet
