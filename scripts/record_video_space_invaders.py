"""Record Space Invaders gameplay videos for all discovered checkpoints.

Auto-discovers checkpoints under models/*SpaceInvaders*/, records one episode
per checkpoint with an annotated score banner, and saves MP4s to
videos/space_invaders/.  Pass --checkpoints to override auto-discovery.

Usage examples
--------------
# Record everything found in models/
python scripts/record_video_space_invaders.py

# Only DQN checkpoints
python scripts/record_video_space_invaders.py --algo dqn

# Specific files
python scripts/record_video_space_invaders.py \
    --checkpoints models/dqn_si_default_SpaceInvaders_v5/dqn_5000000_steps.zip \
                  models/dqn_si_default_SpaceInvaders_v5/dqn_final.zip \
    --checkpoint_algo dqn
"""
import os
import re
import glob
import argparse
import numpy as np
import gymnasium as gym
import ale_py
import imageio
from PIL import Image, ImageDraw
from stable_baselines3 import DQN, A2C, PPO
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
ALGO_MAP = {"dqn": DQN, "a2c": A2C, "ppo": PPO}
# One banner pixel row per point of score keeps things readable at native resolution
BANNER_HEIGHT = 22
BANNER_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 220, 0)


def _annotate(frame: np.ndarray, label: str, score: float) -> np.ndarray:
    """Prepend a thin black banner showing the run label and current score."""
    img = Image.fromarray(frame)
    banner = Image.new("RGB", (img.width, BANNER_HEIGHT), color=BANNER_COLOR)
    draw = ImageDraw.Draw(banner)
    draw.text((4, 5), f"{label}   score: {int(score)}", fill=TEXT_COLOR)
    out = Image.new("RGB", (img.width, img.height + BANNER_HEIGHT))
    out.paste(banner, (0, 0))
    out.paste(img, (0, BANNER_HEIGHT))
    return np.array(out)


def record_episode(
    algo: str,
    checkpoint_path: str,
    label: str,
    output_path: str,
    fps: int = 30,
) -> float:
    AlgoClass = ALGO_MAP[algo]

    # frameskip=1 disables the env's built-in frame skip so AtariPreprocessing
    # can apply its own without doubling up (raises ValueError otherwise)
    env = gym.make(ENV_ID, render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, stack_size=4)

    # Load without a vectorized env; predict() works on a single (C,H,W) obs
    model = AlgoClass.load(checkpoint_path)

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    done = False

    while not done:
        raw = env.render()
        if raw is not None:
            frames.append(_annotate(raw, label, total_reward))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"  saved {output_path}  ({len(frames)} frames, score={int(total_reward)})")
    return total_reward


def _step_label(algo: str, step: int | None) -> str:
    if step is None:
        return f"{algo}_final"
    if step >= 1_000_000:
        n = step / 1_000_000
        return f"{algo}_{n:.0f}M" if n == int(n) else f"{algo}_{n:.1f}M"
    return f"{algo}_{step // 1_000}k"


def discover_checkpoints(models_dir: str, algo_filter: str | None) -> list[dict]:
    """Glob models/*SpaceInvaders*/*.zip and return sorted metadata dicts."""
    entries = []
    for path in glob.glob(os.path.join(models_dir, "*SpaceInvaders*", "*.zip")):
        # Skip best-model saves (they live in a 'best/' subdir)
        if os.sep + "best" + os.sep in path or "/best/" in path:
            continue

        dirname = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)

        algo = next((a for a in ("dqn", "a2c", "ppo") if dirname.lower().startswith(a)), None)
        if algo is None:
            continue
        if algo_filter and algo != algo_filter:
            continue

        m = re.search(r"(\d+)_steps", fname)
        step = int(m.group(1)) if m else None
        sort_key = step if step is not None else int(1e12)

        entries.append({
            "algo": algo,
            "step": step,
            "sort_key": sort_key,
            "label": _step_label(algo, step),
            "path": path,
        })

    return sorted(entries, key=lambda e: (e["algo"], e["sort_key"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record Space Invaders videos from trained checkpoints"
    )
    parser.add_argument(
        "--models_dir", default="models",
        help="Root directory containing trained model subdirs (default: models)",
    )
    parser.add_argument(
        "--output_dir", default="videos/space_invaders",
        help="Destination directory for MP4 files (default: videos/space_invaders)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument(
        "--algo", choices=["dqn", "a2c", "ppo"], default=None,
        help="Restrict auto-discovery to one algorithm",
    )
    # Manual override
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Explicit checkpoint paths; skips auto-discovery",
    )
    parser.add_argument(
        "--checkpoint_algo", choices=["dqn", "a2c", "ppo"],
        help="Algorithm class for --checkpoints mode (required when using --checkpoints)",
    )
    args = parser.parse_args()

    if args.checkpoints:
        if not args.checkpoint_algo:
            parser.error("--checkpoint_algo is required when using --checkpoints")
        entries = []
        for i, path in enumerate(args.checkpoints):
            m = re.search(r"(\d+)_steps", os.path.basename(path))
            step = int(m.group(1)) if m else None
            entries.append({
                "algo": args.checkpoint_algo,
                "step": step,
                "label": _step_label(args.checkpoint_algo, step),
                "path": path,
            })
    else:
        entries = discover_checkpoints(args.models_dir, args.algo)

    if not entries:
        print(
            "No Space Invaders checkpoints found.\n"
            "  • Train first with:  python scripts/train_dqn.py\n"
            "  • Or pass explicit paths via --checkpoints"
        )
        return

    print(f"Recording {len(entries)} video(s) → {args.output_dir}/\n")
    summary = []

    for entry in entries:
        print(f"[{entry['algo'].upper()}] {entry['label']}")
        out_path = os.path.join(args.output_dir, f"{entry['label']}.mp4")
        score = record_episode(entry["algo"], entry["path"], entry["label"], out_path, fps=args.fps)
        summary.append({**entry, "score": score, "video": out_path})

    # Summary table
    col = 22
    print("\n" + "=" * 68)
    print(f"{'Algo':<6} {'Label':<{col}} {'Score':>7}  File")
    print("-" * 68)
    for row in summary:
        print(f"{row['algo'].upper():<6} {row['label']:<{col}} {int(row['score']):>7}  {row['video']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
