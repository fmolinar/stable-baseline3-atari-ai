@echo off
cd /d "D:\Fresno State\AI\Deep-Reinforcement"
call venv\Scripts\activate
echo Starting A2C training on Pong...
start "A2C Pong Training" cmd /k "python scripts/train_a2c.py --env ALE/Pong-v5 --experiment a2c_default --save_dir models --config configs/a2c_config.yaml"
echo Training window launched. You can close this window.
timeout /t 3
