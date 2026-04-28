# Experiment Results — Concise Summary

> One-page-style summary covering environments, algorithms, hyperparameters
> explored, saved checkpoints, evaluation scores, and short observations.
> Detailed write-up: [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Environments

| ID | Score range |
|---|---|
| `ALE/Pong-v5` | −21 to +21 |
| `ALE/SpaceInvaders-v5` | 0 to ~3000+ |

## Algorithms

| Algorithm | Family | SB3 class |
|---|---|---|
| DQN | Off-policy, value-based | `stable_baselines3.DQN` |
| A2C | On-policy actor-critic | `stable_baselines3.A2C` |
| PPO | On-policy clipped policy gradient | `stable_baselines3.PPO` |

All runs use `CnnPolicy`, `make_atari_env`, `VecFrameStack(n_stack=4)`. Total
training budget: **10 M timesteps** per (algo, env).

## Hyperparameters Explored

Defined in [`configs/dqn_config.yaml`](../configs/dqn_config.yaml),
[`configs/a2c_config.yaml`](../configs/a2c_config.yaml),
[`configs/ppo_config.yaml`](../configs/ppo_config.yaml).

**DQN sweeps:** learning_rate ∈ {1e-4, 5e-4, 1e-3}, buffer_size ∈ {100k, 500k,
1M}, batch_size ∈ {32, 64}, gamma ∈ {0.95, 0.99}, target_update_interval ∈
{1k, 2k, 5k, 10k}, exploration_fraction ∈ {0.1, 0.15, 0.2}.

**A2C sweeps:** learning_rate ∈ {3e-4, 7e-4}, n_envs ∈ {4, 8},
n_steps ∈ {5, 10}, ent_coef ∈ {0.001, 0.01, 0.02, 0.05}.

**PPO sweeps:** n_envs ∈ {4, 8}, n_steps ∈ {128, 256, 512}, batch_size ∈
{64, 256, 512}, ent_coef ∈ {0.001, 0.01, 0.03}, clip_range = 0.1.

## Saved Checkpoints

Per run, every 100 k (Pong) or 500 k (Space Invaders) timesteps, plus a
`*_final.zip` at termination. Naming pattern:
`models/<experiment>_<env>/<algo>_<step>_steps.zip`.

| Env | Steps captured |
|---|---|
| Pong (10 M) | 100k, 200k, …, 10M, final |
| Space Invaders (10 M) | 100k, 500k, 1M, 2M, 5M, 10M, final |

Only `.zip` checkpoints are excluded from git (size). Full evaluation
trajectories per checkpoint are stored as JSON in [`results/`](../results/).

## Evaluation Scores

20 held-out deterministic episodes per checkpoint.

### Space Invaders

| Algo | 100k | 500k | 1M | 2M | 5M | 10M (final) |
|---|---|---|---|---|---|---|
| DQN | 217 ± 123 | 365 ± 127 | 464 ± 152 | 600 ± 167 | 897 ± 291 | **1064 ± 485** |
| A2C | 363 ± 213 | 411 ± 134 | 319 ± 143 | 520 ± 147 | 613 ± 210 | **914 ± 370** |
| PPO | 142 ± 40  | 540 ± 171 | 515 ± 200 | 509 ± 179 | 729 ± 183 | **861 ± 294** |

### Pong

| Algo | 100k | 1M | 2M | 5M | 7M | 10M (final) |
|---|---|---|---|---|---|---|
| DQN | −21.0 ± 0.2 | −16.2 ± 2.2 | −6.0 ± 6.2 | +5.3 ± 4.7 | +7.8 ± 5.7 | **+9.2 ± 4.7** |
| A2C | −21.0 ± 0.0 | −19.9 ± 1.1 | −17.5 ± 1.9 | −12.5 ± 3.2 | −15.4 ± 2.4 | **−8.4 ± 6.2** |
| PPO | −21.0 ± 0.0 | −18.8 ± 2.0 | −16.1 ± 2.8 | −7.1 ± 4.4 | −7.3 ± 5.7 | **−3.1 ± 6.3** |

## Short Observations

- **DQN wins both envs at 10 M.** Decisive on Pong, narrow on Space Invaders.
- **On-policy methods are competitive on the harder env.** A2C/PPO finish within
  ~25% of DQN on Space Invaders.
- **Hyperparameter sensitivity is concentrated in the first 1–2 M steps.** LR
  and entropy decide whether the agent breaks out of the random-policy floor.
- **Resumed training is seamless** — `Algo.load(..., env=…)` plus
  `reset_num_timesteps=False` continues the curve with no regression.
- **High reward variance ≠ unstable learning.** Space Invaders' 20-episode std
  is large because the score scale itself is noisy (UFO bonuses, life count),
  not because the policy oscillates.
