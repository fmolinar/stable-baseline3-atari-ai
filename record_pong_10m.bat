@echo off
title Record Pong 10M Videos
cd /d "%~dp0"
echo ============================================================
echo  Recording Pong 10M Training Videos
echo  Checkpoints: early (500k), mid (5M), final
echo  Algorithms : PPO, DQN, A2C
echo ============================================================
echo.

if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo ============================================
echo  PPO - Pong 10M
echo ============================================

echo [1/3] PPO - early (500k steps)...
python scripts/record_video.py --algo ppo --env ALE/Pong-v5 ^
    --checkpoints models/ppo_10m_Pong_v5/ppo_500000_steps.zip ^
    --labels early ^
    --output_dir videos/ppo_10m_pong

echo [2/3] PPO - mid (5M steps)...
python scripts/record_video.py --algo ppo --env ALE/Pong-v5 ^
    --checkpoints models/ppo_10m_Pong_v5/ppo_5000000_steps.zip ^
    --labels mid ^
    --output_dir videos/ppo_10m_pong

echo [3/3] PPO - final...
python scripts/record_video.py --algo ppo --env ALE/Pong-v5 ^
    --checkpoints models/ppo_10m_Pong_v5/ppo_final.zip ^
    --labels final ^
    --output_dir videos/ppo_10m_pong

echo.
echo ============================================
echo  DQN - Pong 10M
echo ============================================

echo [1/3] DQN - early (500k steps)...
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 ^
    --checkpoints models/dqn_10m_Pong_v5/dqn_500000_steps.zip ^
    --labels early ^
    --output_dir videos/dqn_10m_pong

echo [2/3] DQN - mid (5M steps)...
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 ^
    --checkpoints models/dqn_10m_Pong_v5/dqn_5000000_steps.zip ^
    --labels mid ^
    --output_dir videos/dqn_10m_pong

echo [3/3] DQN - final...
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 ^
    --checkpoints models/dqn_10m_Pong_v5/dqn_final.zip ^
    --labels final ^
    --output_dir videos/dqn_10m_pong

echo.
echo ============================================
echo  A2C - Pong 10M
echo ============================================

echo [1/3] A2C - early (500k steps)...
python scripts/record_video.py --algo a2c --env ALE/Pong-v5 ^
    --checkpoints models/a2c_10m_Pong_v5/a2c_500000_steps.zip ^
    --labels early ^
    --output_dir videos/a2c_10m_pong

echo [2/3] A2C - mid (5M steps)...
python scripts/record_video.py --algo a2c --env ALE/Pong-v5 ^
    --checkpoints models/a2c_10m_Pong_v5/a2c_5000000_steps.zip ^
    --labels mid ^
    --output_dir videos/a2c_10m_pong

echo [3/3] A2C - final...
python scripts/record_video.py --algo a2c --env ALE/Pong-v5 ^
    --checkpoints models/a2c_10m_Pong_v5/a2c_final.zip ^
    --labels final ^
    --output_dir videos/a2c_10m_pong

echo.
echo ============================================================
echo  All recordings done! Videos saved to:
echo    videos\ppo_10m_pong\ppo_Pong_v5_early.mp4
echo    videos\ppo_10m_pong\ppo_Pong_v5_mid.mp4
echo    videos\ppo_10m_pong\ppo_Pong_v5_final.mp4
echo    videos\dqn_10m_pong\dqn_Pong_v5_early.mp4
echo    videos\dqn_10m_pong\dqn_Pong_v5_mid.mp4
echo    videos\dqn_10m_pong\dqn_Pong_v5_final.mp4
echo    videos\a2c_10m_pong\a2c_Pong_v5_early.mp4
echo    videos\a2c_10m_pong\a2c_Pong_v5_mid.mp4
echo    videos\a2c_10m_pong\a2c_Pong_v5_final.mp4
echo ============================================================
pause
