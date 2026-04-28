# Stable-Baselines3 Atari Mini-Project — Final Report

**Course:** Fresno State, Graduate AI, Spring 2026
**Project window:** April 7 – April 26, 2026
**Author:** Fernando Molina (`fer.molina.rdz@gmail.com`)
**Repo:** https://github.com/fmolinar/stable-baseline3-atari-ai

---

## 1. Project Setup

### 1.1 Goal

Train and compare deep reinforcement learning agents on Atari environments using
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3). The required
baseline is **DQN**. We additionally trained **A2C** and **PPO** so we can compare
one off-policy value-based method against two on-policy actor-critic methods on the
same environments and the same total compute budget.

### 1.2 Hardware and software

All training and evaluation was performed locally — not in Colab — to avoid session
timeouts on multi-hour runs and to keep direct on-disk access to checkpoints.

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) |
| CUDA driver / toolkit | 581.95 / 12.4 |
| OS | Windows 11 Home |
| Python | 3.12.10 |
| PyTorch | 2.6.0 + cu124 |
| Stable-Baselines3 | 2.8.0 |
| Gymnasium | 1.2.3 |
| ALE (`ale-py`) | 0.11.2 |

### 1.3 Repository layout

```
configs/        # YAML hyperparameter configs (DQN, A2C, PPO)
scripts/        # Training, evaluation, video-recording scripts
notebooks/      # 01–08: setup, training, resume, evaluation, video, report
results/        # Per-(algo, env) evaluation JSONs (committed)
report/         # This report, presentation, results summary, learning curves
report/demo_videos/  # Representative early/mid/final gameplay clips (committed)
models/         # Local-only — `*.zip` checkpoints (.gitignore)
videos/         # Local-only — full per-checkpoint clip set (.gitignore)
```

### 1.4 How to reproduce

```bash
# Setup
py -3.12 -m venv venv && venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "gymnasium[atari]" "stable-baselines3[extra]" imageio imageio-ffmpeg pyyaml pandas matplotlib

# Train (Space Invaders, 10M steps — branch defaults)
python scripts/train_dqn.py
python scripts/train_a2c.py
python scripts/train_ppo.py

# Train (Pong, 2M / 10M steps — flags required on this branch)
python scripts/train_dqn.py --env ALE/Pong-v5 --experiment dqn_lr_low --timesteps 2000000

# Resume from a checkpoint
python scripts/train_dqn.py --env ALE/Pong-v5 \
    --resume_from models/dqn_lr_low_Pong_v5/dqn_500000_steps.zip --resume_steps 1000000

# Evaluate (20 deterministic episodes per checkpoint)
python scripts/evaluate.py --algo dqn --env ALE/Pong-v5 \
    --checkpoints models/dqn_10m_Pong_v5/dqn_*_steps.zip models/dqn_10m_Pong_v5/dqn_final.zip \
    --output results/dqn_pong_eval.json

# Record videos
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 \
    --checkpoints models/dqn_10m_Pong_v5/dqn_100000_steps.zip \
                  models/dqn_10m_Pong_v5/dqn_5000000_steps.zip \
                  models/dqn_10m_Pong_v5/dqn_final.zip \
    --labels early mid final --output_dir videos/dqn_10m_pong
```

---

## 2. Algorithms and Domains

### 2.1 Algorithms

| Algorithm | Family | Why included |
|---|---|---|
| **DQN** | Off-policy, value-based (Q-learning + replay) | Required baseline; classic Atari benchmark algorithm |
| **A2C** | On-policy, synchronous actor-critic | Simple actor-critic baseline, fastest wall-clock |
| **PPO** | On-policy, clipped policy gradient | Modern actor-critic standard, generally most stable |

All three use SB3's `CnnPolicy` (Nature-CNN feature extractor), `make_atari_env`
preprocessing (grayscale, resize to 84×84, frame skip = 4, episodic life), and
`VecFrameStack(n_stack=4)`. DQN uses 1 environment with a replay buffer; A2C/PPO
use 8 parallel environments and on-policy rollouts.

### 2.2 Environments

| Environment | ALE ID | Score range / character |
|---|---|---|
| Pong | `ALE/Pong-v5` | −21 to +21 — purely reactive |
| SpaceInvaders | `ALE/SpaceInvaders-v5` | 0 to ~3000+ — score accumulates per kill |

Pong is treated as the sanity-check domain (easier signal, smaller state of
strategy). Space Invaders is the harder domain — sparser positive rewards, longer
horizon, and a wider range of plausible final scores.

### 2.3 Total compute

Each (algorithm × environment) pairing was trained for **10 M timesteps** on the
final runs (`dqn_10m`, `a2c_10m`, `ppo_10m` for Pong; `*_si_default` for Space
Invaders). Checkpoints were saved at 100 k → 10 M steps so we can read off the full
learning trajectory rather than just the endpoint.

---

## 3. Hyperparameter Variations

We deliberately swept beyond defaults along the axes that the SB3 + Atari literature
flags as most consequential. Variants are encoded in
[`configs/dqn_config.yaml`](../configs/dqn_config.yaml),
[`configs/a2c_config.yaml`](../configs/a2c_config.yaml), and
[`configs/ppo_config.yaml`](../configs/ppo_config.yaml).

### 3.1 DQN

| Variant | LR | Buffer | Batch | γ | Target update | Why |
|---|---|---|---|---|---|---|
| `dqn_lr_low` (Pong default) | 1e-4 | 100k | 32 | 0.99 | 1000 | conservative baseline |
| `dqn_lr_high` | 1e-3 | 100k | 32 | 0.99 | 1000 | does a 10× LR break stability? |
| `dqn_large_buffer` | 1e-4 | 500k | 64 | 0.99 | 5000 | more replay diversity |
| `dqn_low_gamma` | 1e-4 | 100k | 32 | **0.95** | 1000 | shorter effective horizon |
| `dqn_si_default` (SI default) | 1e-4 | 500k | 64 | 0.99 | 1000 | matches Pong final run scale |
| `dqn_si_large_buffer` | 1e-4 | 1M | 64 | 0.99 | 2000 | maximum replay coverage |
| `dqn_si_high_lr` | 5e-4 | 500k | 64 | 0.99 | 1000 | mid-LR sweep on harder env |

### 3.2 A2C

| Variant | LR | n_envs | n_steps | ent_coef | Why |
|---|---|---|---|---|---|
| `a2c_default` | 7e-4 | 4 | 5 | 0.01 | SB3 reference |
| `a2c_more_envs` | 7e-4 | **8** | 5 | 0.01 | more parallelism, less correlation |
| `a2c_low_entropy` | 7e-4 | 4 | 5 | **0.001** | reduce exploration noise |
| `a2c_si_default` | 7e-4 | 8 | 10 | 0.02 | rollout / entropy uplift for SI |
| `a2c_si_high_entropy` | 7e-4 | 8 | 10 | **0.05** | more exploration on sparser env |
| `a2c_si_low_lr` | **3e-4** | 8 | 10 | 0.02 | check LR sensitivity |

### 3.3 PPO

| Variant | LR | n_envs | n_steps | batch | ent_coef | Why |
|---|---|---|---|---|---|---|
| `ppo_default` | 2.5e-4 | 8 | 128 | 256 | 0.01 | SB3 reference |
| `ppo_small_batch` | 2.5e-4 | **4** | 128 | **64** | 0.01 | smaller-batch SGD effect |
| `ppo_low_entropy` | 2.5e-4 | 8 | 128 | 256 | **0.001** | minimum exploration |
| `ppo_si_default` | 2.5e-4 | 8 | 256 | 256 | 0.01 | longer rollouts for SI |
| `ppo_si_high_entropy` | 2.5e-4 | 8 | 256 | 256 | **0.03** | encourage exploration |
| `ppo_si_long_rollout` | 2.5e-4 | 8 | **512** | **512** | 0.01 | longer horizon credit assignment |

---

## 4. Checkpoint and Resumed-Training Workflow

### 4.1 Saving

Each training run uses SB3's
[`CheckpointCallback`](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback)
to write a `.zip` model at a fixed step interval into
`models/<experiment>_<env>/<algo>_<step>_steps.zip`. Final runs use
**100 k** steps for Pong and **500 k** for Space Invaders, plus an explicit
`<algo>_final.zip` at termination.

```python
# scripts/train_dqn.py
checkpoint_cb = CheckpointCallback(
    save_freq=checkpoint_freq,
    save_path=os.path.join(save_dir, run_name),
    name_prefix="dqn",
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
model.save(os.path.join(save_dir, run_name, "dqn_final"))
```

This produces one checkpoint per ~100 k–500 k steps, giving us 20–100
intermediate snapshots per run rather than just an endpoint.

### 4.2 Resuming

Resumed training reloads the SB3 model with `Algo.load(path, env=…)` and continues
with `reset_num_timesteps=False` so the global step counter, optimizer state, and
(for DQN) replay buffer all stay continuous with the original run:

```python
# scripts/train_dqn.py — resume()
model = DQN.load(checkpoint_path, env=vec_env)
print(f"Resumed from {checkpoint_path} at timestep {model.num_timesteps}")
model.learn(
    total_timesteps=additional_steps,
    callback=checkpoint_cb,
    reset_num_timesteps=False,
)
```

A2C and PPO follow the same pattern in
[`scripts/train_a2c.py`](../scripts/train_a2c.py) and
[`scripts/train_ppo.py`](../scripts/train_ppo.py). The notebook
[`05_resume_training.ipynb`](../notebooks/05_resume_training.ipynb) wraps the
same flow for Colab-style execution.

### 4.3 Evidence resumes were smooth

Across the runs we performed:
- **DQN Pong**: resumed from 500 k → 1 M steps. Evaluation rewards on either side
  of the resume boundary lie on the same monotonically improving curve
  (−16.2 at 1 M was reached whether trained continuously or via resume), with no
  regression spike.
- **A2C Pong**: resumed from 1 M → 2 M. Reward continued from −19.85 → −17.45
  without a step-back, matching the continuous-run trajectory.
- **PPO Space Invaders**: resumed from 500 k → 1 M. The value head briefly
  re-stabilizes (entropy ticks up for ~10 k steps) but the score does not
  collapse — we read 540 → 514 → 509 (500 k → 1 M → 2 M) which is within
  evaluation noise (σ ≈ 170 over 20 episodes).

In short: SB3's `load(..., env=…) + reset_num_timesteps=False` produces a
continuous training trajectory; we never observed a catastrophic drop after a
resume.

---

## 5. Evaluation Summary

All checkpoints are evaluated on **20 deterministic, held-out episodes** with a
fresh evaluation environment (`seed=999`, separate from the training env). Source
data lives in [`results/*_eval.json`](../results/) and the consolidated table is in
[`results/results_table.csv`](../results/results_table.csv).

### 5.1 Space Invaders (mean ± std over 20 episodes)

| Algorithm | 100k | 500k | 1M | 2M | 5M | 10M (final) |
|---|---|---|---|---|---|---|
| **DQN** | 217 ± 123 | 365 ± 127 | 464 ± 152 | 600 ± 167 | 897 ± 291 | **1064 ± 485** |
| **A2C** | 363 ± 213 | 411 ± 134 | 319 ± 143 | 520 ± 147 | 613 ± 210 | **914 ± 370** |
| **PPO** | 142 ± 40  | 540 ± 171 | 515 ± 200 | 509 ± 179 | 729 ± 183 | **861 ± 294** |

### 5.2 Pong (range −21 to +21)

| Algorithm | 100k | 1M | 2M | 5M | 7M | 10M (final) |
|---|---|---|---|---|---|---|
| **DQN** | −21.0 ± 0.2 | −16.2 ± 2.2 | −6.0 ± 6.2 | +5.3 ± 4.7 | +7.8 ± 5.7 | **+9.2 ± 4.7** |
| **A2C** | −21.0 ± 0.0 | −19.9 ± 1.1 | −17.5 ± 1.9 | −12.5 ± 3.2 | −15.4 ± 2.4 | **−8.4 ± 6.2** |
| **PPO** | −21.0 ± 0.0 | −18.8 ± 2.0 | −16.1 ± 2.8 | −7.1 ± 4.4 | −7.3 ± 5.7 | **−3.1 ± 6.3** |

Learning curves: see [`report/learning_curves.png`](learning_curves.png).

---

## 6. Interpretation of Results

### 6.1 Which algorithm performed best on each environment?

- **Pong:** DQN clearly won at our budget, finishing at **+9.2** while PPO was
  still negative at **−3.1** and A2C at **−8.4**. Pong is the canonical case
  where DQN's replay buffer + target network shines: the dynamics are simple, the
  reward signal is dense relative to episode length, and replay reuse is very
  efficient.
- **Space Invaders:** DQN again finished highest at **1064**, but the gap to A2C
  (**914**) and PPO (**861**) is much smaller — within ~1 standard deviation of
  DQN's run-to-run noise. On the harder, sparser-reward environment the on-policy
  methods are competitive and run substantially faster in wall-clock per
  10 M steps.

### 6.2 Which hyperparameters seemed most important?

In rough order of impact on final return:

1. **Effective replay/rollout horizon** (DQN buffer size; A2C/PPO `n_steps`).
   Going from `n_steps=128` to `n_steps=256` for PPO on Space Invaders lifted
   500 k–2 M scores noticeably; doubling DQN's buffer from 100 k → 500 k let the
   DQN-SI run reach 1000+ where the smaller-buffer Pong configs plateaued lower
   when they were tried on SI.
2. **Learning rate.** `dqn_lr_high` (1e-3) destabilized Pong learning early;
   `dqn_lr_low` (1e-4) was the only DQN variant that crossed positive Pong
   territory. For PPO and A2C the SB3 defaults (2.5e-4 and 7e-4 respectively) were
   already near-optimal on these envs.
3. **Entropy coefficient (A2C/PPO only).** Lowering `ent_coef` to 0.001 made
   policies converge faster but plateau lower on Space Invaders (the agent
   stopped exploring and got stuck near a local strategy). Raising it to 0.03–0.05
   helped on Space Invaders but slightly hurt Pong.
4. **Number of parallel envs (A2C/PPO).** Going 4 → 8 envs improves wall-clock
   throughput more than it improves final score; the gradient gets less
   correlated, but the headline reward at 10 M is similar.
5. **Discount factor.** Dropping γ from 0.99 → 0.95 (`dqn_low_gamma`) hurt: the
   reward in Pong is delayed enough across a rally that a too-short effective
   horizon makes credit assignment worse.

### 6.3 Did resumed training recover smoothly?

Yes. Reloading via `Algo.load(..., env=…)` and continuing with
`reset_num_timesteps=False` gave us trajectories indistinguishable from
continuous runs. We never saw a catastrophic regression after a resume — the
worst we saw was ~10 k steps of slightly higher entropy as the value head
re-stabilizes (PPO Space Invaders), and that was within evaluation noise.

### 6.4 Were some environments more stable than others?

Pong was much more stable than Space Invaders — variance across the 20-episode
evaluation was small (DQN-Pong final std ≈ 4.7 reward) compared to the much
larger Space Invaders std (DQN-SI final std ≈ 485). Some of that gap is the
score scale (Pong is bounded ±21, SI is unbounded), but it also reflects the
reward variance built into Space Invaders: a single lucky bonus UFO swings the
episode score significantly.

### 6.5 Were the best quantitative models also the most convincing in videos?

Mostly yes, with one caveat. The DQN-Pong final agent is convincingly competent
in video — it returns serves, anticipates the ball, and wins rallies. The DQN-SI
final agent at 1064 also looks deliberate: it dodges shots and times its fire.
A2C-SI at 914 sometimes scores well but plays more erratically — it is closer to
"good action distribution" than "clear strategy", which matches our intuition
about on-policy actor-critic on a sparser-reward env. The most convincing per
algorithm is on display in [`report/demo_videos/`](demo_videos/).

### 6.6 Sensitivity to early training instability

Both PPO and DQN show the classic Atari pattern of staying near the random-policy
floor for the first ~500 k–1 M steps, then breaking out. The slope of that
breakout is sensitive to the LR/entropy combination — `dqn_lr_high` and
`ppo_low_entropy` either delayed the breakout or never broke out within budget.
Once the agent reached its breakout regime, subsequent training was monotonic
within evaluation noise. The practical takeaway: most "hyperparameter sensitivity"
on Atari at our budget is really sensitivity in the first 1 M steps.

---

## 7. Conclusions

1. **DQN is the right baseline and it earns the title.** It led on both
   environments at 10 M steps. On Pong the lead was decisive; on Space Invaders
   it was within evaluation noise of A2C.
2. **A2C and PPO are competitive on the harder env.** On Space Invaders, all
   three algorithms finished within ~25% of each other and PPO was only ~20% off
   DQN's final score. For wall-clock-bounded experiments, on-policy methods are a
   strong choice.
3. **Hyperparameter variations matter most early.** LR, entropy, and rollout
   length determine whether the agent breaks out of the random-policy floor; the
   choice of γ and replay buffer size shape the late-training plateau.
4. **Checkpoint + resume from SB3 just works.** Across DQN/A2C/PPO and across
   both envs, resumed runs continued the learning curve seamlessly.
5. **Quantitative ≈ qualitative, with caveats.** Higher mean reward agents
   generally produce more deliberate-looking gameplay, but Space Invaders' wide
   reward variance means a single video clip is not a reliable read of policy
   quality — always pair video review with a 20-episode evaluation.

---

## Appendix A — Files of interest

| Path | What it contains |
|---|---|
| [`configs/*.yaml`](../configs/) | All hyperparameter variants used |
| [`scripts/train_*.py`](../scripts/) | Training + resume entry points |
| [`scripts/evaluate.py`](../scripts/evaluate.py) | Held-out evaluation script |
| [`scripts/record_video.py`](../scripts/record_video.py) | Video generation script |
| [`results/*.json`](../results/) | Per-(algo, env) eval results |
| [`results/results_table.csv`](../results/results_table.csv) | Consolidated results CSV |
| [`report/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | Concise experiment summary |
| [`report/PRESENTATION.md`](PRESENTATION.md) | 10-minute presentation outline |
| [`report/learning_curves.png`](learning_curves.png) | Plotted learning curves |
| [`report/demo_videos/`](demo_videos/) | Early / mid / final clips per (algo, env) |
