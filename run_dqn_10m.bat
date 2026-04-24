@echo off
title DQN 10M Pong Training
cd /d "%~dp0"
echo ============================================================
echo  DQN 10M Pong Training
echo  Experiment : dqn_10m
echo  Env        : ALE/Pong-v5
echo  Timesteps  : 10,000,000
echo  Checkpoints: every 50,000 steps
echo  Output     : models\dqn_10m_Pong_v5\
echo ============================================================
echo.
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)
python scripts/train_dqn.py ^
    --env ALE/Pong-v5 ^
    --config configs/dqn_config.yaml ^
    --experiment dqn_10m ^
    --save_dir models
echo.
echo DQN Training finished (or exited). Press any key to close.
pause
