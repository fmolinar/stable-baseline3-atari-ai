# Demo Videos

A representative subset of gameplay clips for the report and presentation. The
full per-checkpoint clip set (every 100 k–500 k steps) is generated locally with
[`scripts/record_video.py`](../../scripts/record_video.py) and
[`scripts/record_video_space_invaders.py`](../../scripts/record_video_space_invaders.py)
but is **not committed** to git (size).

## Layout

```
pong_10m/
  {dqn,a2c,ppo}_Pong_v5_early.mp4   # ≈ 100k-step checkpoint
  {dqn,a2c,ppo}_Pong_v5_mid.mp4     # mid-training checkpoint
  {dqn,a2c,ppo}_Pong_v5_final.mp4   # 10M / final
space_invaders_10m/
  {dqn,a2c,ppo}_early_100k.mp4      # 100k checkpoint
  {dqn,a2c,ppo}_mid_1M.mp4          # 1M checkpoint
  {dqn,a2c,ppo}_final.mp4           # 10M / final
```

Each filename meets the project's "early checkpoint, later checkpoint, final or
best model" deliverable. Per-frame settings: 30 fps, deterministic policy,
matches training preprocessing (84×84 grayscale, frame skip 4, 4-frame stack).

## Regenerating

```bash
# Pong (10M)
python scripts/record_video.py --algo dqn --env ALE/Pong-v5 \
    --checkpoints models/dqn_10m_Pong_v5/dqn_100000_steps.zip \
                  models/dqn_10m_Pong_v5/dqn_5000000_steps.zip \
                  models/dqn_10m_Pong_v5/dqn_final.zip \
    --labels early mid final --output_dir videos/dqn_10m_pong

# Space Invaders (10M, full per-100k clip set)
python scripts/record_video_space_invaders.py
```
