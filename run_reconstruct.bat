@echo off
chcp 65001 > nul
echo =================================================================
echo Transform music using prompt: prepare the environment
echo =================================================================

:: 1. Check PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    pause
    exit /b
)

:: 2. Create env
if not exist "env" (
    echo Creating local environment...
    python -m venv music_env
)

echo Activating local environment...
call env\Scripts\activate.bat

:: 3. Set mmodels directory
if not exist "E:\Models" (
    echo Creating directory for models E:\Models...
    mkdir "E:\Models"
)
set "HF_HOME=E:\Models"
set "AUDIOCRAFT_CACHE_DIR=E:\Models"
echo The directory for models is set to %HF_HOME%

:: 4. Install components
echo Installing components...

python -m pip install --upgrade pip --quiet

echo   -[1/4] Installing pytorch for CUDA 12.6...
pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo   -[2/4] Installing av...
pip install av --only-binary=:all: --upgrade --quiet

echo   -[3/4] Installing Meta AudioCraft...
pip install audiocraft --no-deps --quiet
pip install omegaconf hydra-core einops local-attention treetable submitit sentencepiece encodec flashy num2words spacy gradio torchmetrics hydra-colorlog librosa scipy --quiet
pip install av julius transformers demucs

echo   -[4/4] Clearing pip installation cache...
pip cache purge --quiet

echo =================================================================
echo Transforming music using GPU
echo =================================================================

set "folder=%~1"
set "user_prompt=%~2"

if "%folder%"=="" (
    set /p folder="1. Drop down the folder with MP3-music you want to transform: "
)
if "%user_prompt%"=="" (
    set /p user_prompt="2. Type a prompt for the desired new style of the music (vocals will not be touched): "
)

set folder=%folder:"=%

echo Reconstructing music...
python "%~dp0reconstruct.py" "%folder%" "%user_prompt%"

pause
