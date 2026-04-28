# 10-Minute Presentation — SB3 Atari Mini-Project

**Presenter:** Fernando Molina
**Date:** April 28 or April 30, 2026
**Time budget:** 10 minutes total — ~1 minute per slide

The slide order below is also the order in which to deliver the talk. Each slide
has a target time, key points, and what to *show on screen*.

---

## Slide 1 — Title (0:30)

- Title: **Stable-Baselines3 Atari Mini-Project**
- Subtitle: DQN vs. A2C vs. PPO on Pong and Space Invaders
- Course / term / name
- One-line goal: *Train DQN as the baseline and benchmark two on-policy
  alternatives across two Atari environments at 10 M timesteps.*

---

## Slide 2 — Project Goal (0:45)

- Required: train DQN on at least one Atari env.
- Beyond required: add **A2C** and **PPO**, run on **two** envs (Pong + Space
  Invaders), at the same 10 M-timestep budget so comparisons are apples-to-apples.
- Save checkpoints throughout training, resume training from a checkpoint,
  evaluate on held-out episodes, and produce gameplay videos.

---

## Slide 3 — Domains and Algorithms (1:00)

| Algorithm | Family | Why included |
|---|---|---|
| DQN | Off-policy, value-based | Required baseline |
| A2C | On-policy actor-critic | Simple AC baseline |
| PPO | On-policy clipped PG | Modern AC standard |

| Environment | Score range | Character |
|---|---|---|
| Pong | −21 to +21 | Reactive, dense reward |
| Space Invaders | 0 to ~3000+ | Sparser, longer-horizon |

All runs use SB3 `CnnPolicy`, Atari preprocessing (84×84 grayscale, frame skip 4),
and frame stacking (n=4).

---

## Slide 4 — Hyperparameter Experiments (1:15)

Highlight the 2–3 most informative sweeps rather than reading every variant.

- **DQN learning rate:** 1e-4 (stable) vs. 1e-3 (destabilizes Pong) — LR matters
  most early in training.
- **DQN replay buffer:** 100k → 500k on Space Invaders unlocked the 1000+
  regime; smaller buffers plateau lower.
- **PPO/A2C entropy:** lowering `ent_coef` to 0.001 collapses exploration on
  Space Invaders; defaults are well-tuned.
- **PPO rollout length:** `n_steps` 128 → 256 helps on the harder env.

Pull the full sweep grid from the report only if asked.

---

## Slide 5 — Checkpoint + Resume Workflow (1:00)

Show a code snippet *and* a learning curve.

```python
# Save during training
checkpoint_cb = CheckpointCallback(save_freq=500_000, save_path=..., name_prefix="dqn")
model.learn(total_timesteps=10_000_000, callback=[checkpoint_cb, eval_cb])
model.save("dqn_final.zip")

# Resume later
model = DQN.load("dqn_500000_steps.zip", env=vec_env)
model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
```

- Checkpoint frequency: 100 k (Pong) / 500 k (Space Invaders).
- Resumed runs continue the learning curve with **no regression** — DQN-Pong
  resumed at 500 k → 1 M lands on the same trajectory as a continuous run.

---

## Slide 6 — Performance Summary, Space Invaders (1:00)

Show the numeric table *and* a Space Invaders learning curve plot.

| Algo | 100k | 1M | 5M | 10M (final) |
|---|---|---|---|---|
| **DQN** | 217 | 464 | 897 | **1064 ± 485** |
| A2C | 363 | 319 | 613 | 914 ± 370 |
| PPO | 142 | 515 | 729 | 861 ± 294 |

**Read:** DQN wins, but A2C and PPO are within ~25% on this harder env.

---

## Slide 7 — Performance Summary, Pong (1:00)

| Algo | 100k | 2M | 5M | 10M (final) |
|---|---|---|---|---|
| **DQN** | −21.0 | −6.0 | +5.3 | **+9.2 ± 4.7** |
| A2C | −21.0 | −17.5 | −12.5 | −8.4 ± 6.2 |
| PPO | −21.0 | −16.1 | −7.1 | −3.1 ± 6.3 |

**Read:** DQN wins decisively on Pong. PPO eventually crosses into the −5 range;
A2C is the slowest to converge.

---

## Slide 8 — Gameplay Clips (1:30)

Play 3 short clips (10–15 s each), narrate as they play.

1. **DQN Pong, early (100k):** moves randomly, loses 0–21.
2. **DQN Pong, final (10M):** returns serves, wins rallies.
3. **DQN Space Invaders, final (10M):** deliberate dodging, focused fire.

Available clips committed under [`report/demo_videos/`](demo_videos/) — early /
mid / final per (algo, env). Mention that the full per-checkpoint clip set
(every 100 k–500 k) is generated locally but not committed (size).

---

## Slide 9 — Major Conclusions (0:45)

- **DQN won on both environments** at the 10 M-step budget.
- **On the harder env, all three algorithms are competitive** — A2C/PPO offer
  better wall-clock time per timestep, which matters if compute is the bottleneck.
- **Hyperparameters matter most in the first 1–2 M steps.** Once an agent
  breaks out of the random-policy floor, the remaining curve is mostly monotone.
- **SB3 checkpoint + resume is reliable** across all three algorithms and both
  environments — no regressions, no special handling required.

---

## Slide 10 — Q&A / Backup (1:15)

Anticipated questions and where the answer lives:

- *"Why local instead of Colab?"* — multi-hour uninterrupted runs, direct
  checkpoint access (Section 1.2).
- *"How did you handle frame skip in videos?"* — disable env's built-in
  `frameskip` (set 1) and let `AtariPreprocessing(frame_skip=4)` apply it once
  ([`scripts/record_video.py:22-28`](../scripts/record_video.py)).
- *"Did any run diverge?"* — `dqn_lr_high` (LR 1e-3) destabilized Pong early;
  `ppo_low_entropy` plateaued lower on Space Invaders.
- *"How robust are the eval numbers?"* — 20 deterministic held-out episodes per
  checkpoint, separate seed (999) from training.

Backup slides available with full hyperparameter grid and learning-curve plots
(see [`report/PROJECT_REPORT.md`](PROJECT_REPORT.md)).

---

## Total time: ~10:00

| Slide | Cumulative time |
|---|---|
| 1. Title | 0:30 |
| 2. Goal | 1:15 |
| 3. Domains + algos | 2:15 |
| 4. Hyperparams | 3:30 |
| 5. Checkpoint + resume | 4:30 |
| 6. SI summary | 5:30 |
| 7. Pong summary | 6:30 |
| 8. Clips | 8:00 |
| 9. Conclusions | 8:45 |
| 10. Q&A | 10:00 |
