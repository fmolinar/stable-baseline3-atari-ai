@echo off
title PPO 10M Space Invaders Training
cd /d "%~dp0"
echo ============================================================
echo  PPO 10M Space Invaders Training
echo  Experiment : ppo_si_default
echo  Env        : ALE/SpaceInvaders-v5
echo  Timesteps  : 10,000,000
echo  Checkpoints: every 500,000 steps
echo  Output     : models\ppo_si_default_SpaceInvaders_v5\
echo ============================================================
echo.
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)
python scripts/train_ppo.py ^
    --env ALE/SpaceInvaders-v5 ^
    --config configs/ppo_config.yaml ^
    --experiment ppo_si_default ^
    --save_dir models
echo.
echo PPO Training finished (or exited). Press any key to close.
pause
