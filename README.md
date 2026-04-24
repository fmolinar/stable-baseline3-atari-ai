# Stable-Baselines3 Atari Mini Project

**Course Project Window:** April 7, 2026 – April 26, 2026  
**Final Submission:** Sunday, April 26, 2026, by 11:59 PM  
**Presentations:** Tuesday, April 28 & Thursday, April 30, 2026

---

## Overview

This project uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) to train and compare reinforcement learning agents on Atari environments. The required baseline algorithm is **DQN**, compared against one or two additional SB3 algorithms (A2C and/or PPO) across one to three Atari domains.

---

## Project Goals

- Train DQN as a required baseline on Atari environments
- Compare DQN with A2C and/or PPO
- Explore multiple hyperparameter configurations
- Save models at multiple training checkpoints
- Resume training from saved checkpoints
- Evaluate trained agents on held-out episodes
- Generate gameplay videos from checkpoints and final models
- Summarize findings with tables, plots, and discussion

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_setup_and_env_test.ipynb       # Environment setup and smoke tests
│   ├── 02_dqn_training.ipynb             # DQN training with checkpoints
│   ├── 03_a2c_training.ipynb             # A2C training with checkpoints
│   ├── 04_ppo_training.ipynb             # PPO training with checkpoints
│   ├── 05_resume_training.ipynb          # Checkpoint loading and resumed training
│   ├── 06_evaluation.ipynb               # Agent evaluation across checkpoints
│   └── 07_video_generation.ipynb         # Gameplay video recording
├── scripts/
│   ├── train_dqn.py                      # DQN training script
│   ├── train_a2c.py                      # A2C training script
│   ├── train_ppo.py                      # PPO training script
│   ├── evaluate.py                       # Evaluation script
│   └── record_video.py                   # Video generation script
├── configs/
│   ├── dqn_config.yaml                   # DQN hyperparameter configs
│   ├── a2c_config.yaml                   # A2C hyperparameter configs
│   └── ppo_config.yaml                   # PPO hyperparameter configs
├── results/
│   └── .gitkeep
├── report/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

---

## Branch Notes: `feature/space-invaders`

> **This branch targets Space Invaders at 10M timesteps.**
> The script defaults have changed from the `main` branch. If you want to train on Pong you must pass the flags explicitly — running a script with no arguments will launch Space Invaders.

| Setting | `main` default | `feature/space-invaders` default |
|---|---|---|
| `--env` | `ALE/Pong-v5` | `ALE/SpaceInvaders-v5` |
| `--timesteps` | 2,000,000 | 10,000,000 |
| `--experiment` (DQN) | `dqn_lr_low` | `dqn_si_default` |
| `--experiment` (A2C) | `a2c_default` | `a2c_si_default` |
| `--experiment` (PPO) | `ppo_default` | `ppo_si_default` |
| Checkpoint frequency | every 100k steps | every 500k steps |
| Eval frequency | every 50k steps | every 250k steps |

**To run Pong on this branch**, pass all three flags explicitly:

```bash
python scripts/train_dqn.py --env ALE/Pong-v5 --experiment dqn_lr_low --timesteps 2000000
python scripts/train_a2c.py --env ALE/Pong-v5 --experiment a2c_default --timesteps 2000000
python scripts/train_ppo.py --env ALE/Pong-v5 --experiment ppo_default --timesteps 2000000
```

Pong models and Space Invaders models are saved to separate directories and will never overwrite each other (paths are derived from `--experiment` + env name, e.g. `models/dqn_lr_low_Pong_v5/` vs `models/dqn_si_default_SpaceInvaders_v5/`).

---

## Environments

| Environment | ALE ID |
|---|---|
| Pong | `ALE/Pong-v5` |
| Breakout | `ALE/Breakout-v5` |
| SpaceInvaders | `ALE/SpaceInvaders-v5` |

---

## Algorithms

| Algorithm | Type | Notes |
|---|---|---|
| DQN | Off-policy, value-based | Required baseline |
| A2C | On-policy, actor-critic | Synchronous advantage actor-critic |
| PPO | On-policy, actor-critic | Proximal policy optimization |

---

## Hyperparameters Explored

**DQN**
- `learning_rate`: 1e-4, 5e-4, 1e-3
- `batch_size`: 32, 64
- `buffer_size`: 10000, 50000, 100000
- `gamma` (discount): 0.99, 0.95
- `target_update_interval`: 1000, 5000, 10000
- `exploration_fraction`: 0.1, 0.2

**A2C / PPO**
- `learning_rate`: 7e-4, 2.5e-4
- `gamma`: 0.99
- `n_envs` (parallel environments): 4, 8
- `n_steps`: 5, 128
- `ent_coef` (entropy coefficient): 0.01, 0.001

---

## Checkpoint Strategy

Models are saved at the following training milestones:

**Pong (2M timesteps)**

| Checkpoint | Timesteps |
|---|---|
| Early | 100,000 |
| Mid | 500,000 |
| Late | 1,000,000 |
| Final | 2,000,000 |

**Space Invaders (10M timesteps)**

| Checkpoint | Timesteps |
|---|---|
| Early | 100,000 |
| | 500,000 |
| | 1,000,000 |
| | 2,000,000 |
| | 5,000,000 |
| Final | 10,000,000 |

---

## Evaluation Protocol

Each checkpoint is evaluated on **20 held-out episodes** with no exploration (deterministic policy). Metrics reported:

- Mean episode reward
- Standard deviation of episode reward
- Mean episode length

---

## Training Hardware

All experiments are trained locally on the following machine:

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) |
| CUDA Driver | 581.95 |
| CUDA Version | 12.4 (via PyTorch cu124 build) |
| OS | Windows 11 Home |
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| Stable-Baselines3 | 2.8.0 |
| Gymnasium | 1.2.3 |
| ALE (ale-py) | 0.11.2 |

Training locally (rather than Google Colab) gives uninterrupted multi-hour runs with no session timeout, direct checkpoint access, and comparable or faster throughput than a Colab free T4 GPU.

---

## Setup

### Local (recommended)

Requires Python 3.12 and a CUDA-capable NVIDIA GPU.

```bash
# 1. Create and activate virtual environment
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install project dependencies
pip install "gymnasium[atari]" "stable-baselines3[extra]" imageio imageio-ffmpeg pyyaml pandas matplotlib

# 4. Verify CUDA is detected
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Google Colab (alternative)

```python
from google.colab import drive
drive.mount("/content/drive")

!pip uninstall -y gym
!pip install -U "gymnasium[atari]" "stable-baselines3[extra]"
```

---

## Quick Start

```bash
# Activate the virtual environment first
venv\Scripts\activate
```

**Space Invaders — 10M steps (branch defaults, no extra flags needed)**

```bash
python scripts/train_dqn.py
python scripts/train_a2c.py
python scripts/train_ppo.py
```

**Pong — 2M steps (must pass flags explicitly on this branch)**

```bash
python scripts/train_dqn.py --env ALE/Pong-v5 --experiment dqn_lr_low --timesteps 2000000
python scripts/train_a2c.py --env ALE/Pong-v5 --experiment a2c_default --timesteps 2000000
python scripts/train_ppo.py --env ALE/Pong-v5 --experiment ppo_default --timesteps 2000000
```

**Resume, evaluate, and record (works for any env)**

```bash
# Resume training from a checkpoint
python scripts/train_dqn.py --env ALE/Pong-v5 --resume_from models/dqn_lr_low_Pong_v5/dqn_500000_steps.zip --resume_steps 1000000

# Evaluate checkpoints
python scripts/evaluate.py --algo dqn --env ALE/Pong-v5 \
    --checkpoints models/dqn_lr_low_Pong_v5/dqn_100000_steps.zip models/dqn_lr_low_Pong_v5/dqn_final.zip

# Record gameplay videos
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 \
    --checkpoints models/dqn_lr_low_Pong_v5/dqn_100000_steps.zip models/dqn_lr_low_Pong_v5/dqn_final.zip \
    --labels early final --output_dir videos
```

Or run the numbered notebooks in order:

```
01 → setup    02 → DQN    03 → A2C    04 → PPO
05 → resume   06 → eval   07 → video
```

---

## Results Summary

> Results will be populated as experiments complete.

| Algorithm | Environment | Checkpoint | Mean Reward | Std |
|---|---|---|---|---|
| DQN | Pong | 100k | TBD | TBD |
| A2C | Pong | 100k | TBD | TBD |
| PPO | Pong | 100k | TBD | TBD |

---

## Notebook Order

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_setup_and_env_test.ipynb` | Install packages, register ALE, smoke test all 3 environments |
| 02 | `02_dqn_training.ipynb` | Train DQN on Pong with checkpoint saving every 100k steps |
| 03 | `03_a2c_training.ipynb` | Train A2C (default + 8-env/low-entropy variant) |
| 04 | `04_ppo_training.ipynb` | Train PPO (default + small-batch variant) |
| 05 | `05_resume_training.ipynb` | Load any checkpoint and continue training |
| 06 | `06_evaluation.ipynb` | Evaluate all checkpoints, build results table and learning curve plot |
| 07 | `07_video_generation.ipynb` | Record early / mid / final gameplay videos as `.mp4` |

---

## What Is and Isn't in This Repo

**Tracked (in git):**
- All Python scripts and notebooks
- Hyperparameter config YAMLs
- `requirements.txt`

**Excluded (local only, in `.gitignore`):**
- `venv/` — virtual environment
- `models/` — trained model checkpoints (`.zip`)
- `videos/` — recorded gameplay (`.mp4`)
- `results/` — evaluation output JSONs
- `logs/` and `tensorboard_logs/`

---

## Team

Fresno State — Graduate AI Course, Spring 2026
