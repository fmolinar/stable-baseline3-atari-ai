@echo off
title Compare Pong 10M vs HuggingFace Pretrained
cd /d "%~dp0"
echo ============================================================
echo  Comparing your Pong 10M models vs SB3 Zoo (HuggingFace)
echo  20 episodes each, env: ALE/Pong-v5
echo ============================================================
echo.

if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Checking for huggingface_sb3...
python -c "import huggingface_sb3" 2>nul
if errorlevel 1 (
    echo Installing huggingface_sb3...
    pip install huggingface_sb3
)

echo.
python scripts/compare_pretrained.py --episodes 20
pause
