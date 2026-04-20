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

| Checkpoint | Timesteps |
|---|---|
| Early | 100,000 |
| Mid | 500,000 |
| Late | 1,000,000 |
| Final | 2,000,000 |

---

## Evaluation Protocol

Each checkpoint is evaluated on **20 held-out episodes** with no exploration (deterministic policy). Metrics reported:

- Mean episode reward
- Standard deviation of episode reward
- Mean episode length

---

## Setup

### Google Colab (recommended)

```python
from google.colab import drive
drive.mount("/content/drive")

!pip uninstall -y gym
!pip install -U "gymnasium[atari]" "stable-baselines3[extra]"
```

### Local

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

gym.register_envs(ale_py)

# Create vectorized, preprocessed Atari env
vec_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Train DQN
model = DQN("CnnPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("checkpoints/dqn_pong_100k")
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

## Team

Fresno State — Graduate AI Course, Spring 2026
