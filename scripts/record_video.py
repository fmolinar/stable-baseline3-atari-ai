"""Record gameplay videos from saved model checkpoints."""
import os
import argparse
import numpy as np
import gymnasium as gym
import ale_py
import imageio
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ALGO_MAP = {"dqn": DQN, "a2c": A2C, "ppo": PPO}


def record_episode(algo: str, checkpoint_path: str, env_id: str, output_path: str, fps: int = 30):
    AlgoClass = ALGO_MAP[algo.lower()]

    # frameskip=1 disables the env's built-in frame-skip so AtariPreprocessing
    # can apply its own (default frame_skip=4) without double-skipping.
    env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
    frames = []

    # Wrap manually to match training preprocessing
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = FrameStackObservation(env, stack_size=4)

    # Load model without vectorized env (predict on single obs)
    model = AlgoClass.load(checkpoint_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}  (total reward: {total_reward:.1f}, {len(frames)} frames)")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["dqn", "a2c", "ppo"])
    parser.add_argument("--env", default="ALE/Pong-v5")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths (early, mid, final)")
    parser.add_argument("--labels", nargs="+",
                        help="Labels for each checkpoint (e.g. early mid final)")
    parser.add_argument("--output_dir", default="videos")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    labels = args.labels or [f"ckpt_{i}" for i in range(len(args.checkpoints))]
    env_short = args.env.split("/")[-1].replace("-", "_")

    for ckpt, label in zip(args.checkpoints, labels):
        out = os.path.join(args.output_dir, f"{args.algo}_{env_short}_{label}.mp4")
        record_episode(args.algo, ckpt, args.env, out, fps=args.fps)


if __name__ == "__main__":
    main()
