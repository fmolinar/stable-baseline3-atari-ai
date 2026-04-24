@echo off
cd /d "D:\Fresno State\AI\Deep-Reinforcement"
call venv\Scripts\activate

echo ============================================
echo  Recording DQN gameplay videos (3 episodes)
echo ============================================

echo [1/6] DQN - early (100k steps)...
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 ^
    --checkpoints models/dqn_lr_low_Pong_v5/dqn_100000_steps.zip ^
    --labels early ^
    --output_dir videos/dqn

echo [2/6] DQN - mid (1M steps)...
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 ^
    --checkpoints models/dqn_lr_low_Pong_v5/dqn_1000000_steps.zip ^
    --labels mid ^
    --output_dir videos/dqn

echo [3/6] DQN - final (2M steps)...
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 ^
    --checkpoints models/dqn_lr_low_Pong_v5/dqn_final.zip ^
    --labels final ^
    --output_dir videos/dqn

echo.
echo ============================================
echo  Recording A2C gameplay videos (3 episodes)
echo ============================================

echo [4/6] A2C - early (100k steps)...
python scripts/record_video.py --algo a2c --env ALE/Pong-v5 ^
    --checkpoints models/a2c_default_Pong_v5/a2c_100000_steps.zip ^
    --labels early ^
    --output_dir videos/a2c

echo [5/6] A2C - mid (1M steps)...
python scripts/record_video.py --algo a2c --env ALE/Pong-v5 ^
    --checkpoints models/a2c_default_Pong_v5/a2c_1000000_steps.zip ^
    --labels mid ^
    --output_dir videos/a2c

echo [6/6] A2C - final (2M steps)...
python scripts/record_video.py --algo a2c --env ALE/Pong-v5 ^
    --checkpoints models/a2c_default_Pong_v5/a2c_final.zip ^
    --labels final ^
    --output_dir videos/a2c

echo.
echo ============================================
echo  All recordings done!
echo  Videos saved to:
echo    videos\dqn\dqn_Pong_v5_early.mp4
echo    videos\dqn\dqn_Pong_v5_mid.mp4
echo    videos\dqn\dqn_Pong_v5_final.mp4
echo    videos\a2c\a2c_Pong_v5_early.mp4
echo    videos\a2c\a2c_Pong_v5_mid.mp4
echo    videos\a2c\a2c_Pong_v5_final.mp4
echo ============================================
pause
