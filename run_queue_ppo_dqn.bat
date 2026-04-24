@echo off
title PPO then DQN Queue
cd /d "%~dp0"
echo ============================================================
echo  Sequential Training Queue
echo  1. PPO 10M  ->  models\ppo_10m_Pong_v5\
echo  2. DQN 10M  ->  models\dqn_10m_Pong_v5\
echo  Total: ~20,000,000 timesteps
echo ============================================================
echo.

echo [QUEUE] Starting PPO 10M training...
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python scripts/train_ppo.py ^
    --env ALE/Pong-v5 ^
    --config configs/ppo_config.yaml ^
    --experiment ppo_10m ^
    --save_dir models
echo.
echo [QUEUE] PPO 10M finished. Starting DQN 10M training...
echo.
python scripts/train_dqn.py ^
    --env ALE/Pong-v5 ^
    --config configs/dqn_config.yaml ^
    --experiment dqn_10m ^
    --save_dir models
echo.
echo ============================================================
echo  [QUEUE] All training complete!
echo  PPO model : models\ppo_10m_Pong_v5\ppo_final.zip
echo  DQN model : models\dqn_10m_Pong_v5\dqn_final.zip
echo ============================================================
pause
