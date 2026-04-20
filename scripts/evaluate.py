"""Evaluate saved checkpoints and produce a results table."""
import os
import argparse
import json
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(ale_py)

ALGO_MAP = {"dqn": DQN, "a2c": A2C, "ppo": PPO}


def evaluate_checkpoint(algo: str, checkpoint_path: str, env_id: str, n_episodes: int = 20):
    AlgoClass = ALGO_MAP[algo.lower()]
    eval_env = make_atari_env(env_id, n_envs=1, seed=999)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    model = AlgoClass.load(checkpoint_path, env=eval_env)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    eval_env.close()
    return {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["dqn", "a2c", "ppo"])
    parser.add_argument("--env", default="ALE/Pong-v5")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to checkpoint .zip files to evaluate")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--output", default="results/eval_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []

    for ckpt in args.checkpoints:
        print(f"Evaluating {ckpt} ...")
        metrics = evaluate_checkpoint(args.algo, ckpt, args.env, args.n_episodes)
        metrics["checkpoint"] = ckpt
        metrics["algo"] = args.algo
        metrics["env"] = args.env
        results.append(metrics)
        print(f"  Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
