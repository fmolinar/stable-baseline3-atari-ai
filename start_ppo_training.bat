@echo off
cd /d "D:\Fresno State\AI\Deep-Reinforcement"
call venv\Scripts\activate
echo Starting PPO training on Pong...
start "PPO Pong Training" cmd /k "python scripts/train_ppo.py --env ALE/Pong-v5 --experiment ppo_default --save_dir models --config configs/ppo_config.yaml"
echo Training window launched. You can close this window.
timeout /t 3
