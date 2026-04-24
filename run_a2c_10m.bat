@echo off
title A2C 10M Pong Training
cd /d "%~dp0"
echo ============================================================
echo  A2C 10M Pong Training
echo  Experiment : a2c_10m
echo  Env        : ALE/Pong-v5
echo  Timesteps  : 10,000,000
echo  Checkpoints: every 50,000 steps
echo  Output     : models\a2c_10m_Pong_v5\
echo ============================================================
echo.
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)
python scripts/train_a2c.py ^
    --env ALE/Pong-v5 ^
    --config configs/a2c_config.yaml ^
    --experiment a2c_10m ^
    --save_dir models
echo.
echo Training finished (or exited). Press any key to close.
pause
