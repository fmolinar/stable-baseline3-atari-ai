"""Compare your 10M Pong models against SB3 Zoo pretrained baselines from HuggingFace."""
import argparse
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

gym.register_envs(ale_py)

ALGO_MAP = {"dqn": DQN, "a2c": A2C, "ppo": PPO}

HF_MODELS = [
    {"label": "PPO  (HuggingFace SB3 Zoo)", "algo": "ppo", "repo": "sb3/ppo-PongNoFrameskip-v4", "file": "ppo-PongNoFrameskip-v4.zip"},
    {"label": "DQN  (HuggingFace SB3 Zoo)", "algo": "dqn", "repo": "sb3/dqn-PongNoFrameskip-v4", "file": "dqn-PongNoFrameskip-v4.zip"},
    {"label": "A2C  (HuggingFace SB3 Zoo)", "algo": "a2c", "repo": "sb3/a2c-PongNoFrameskip-v4", "file": "a2c-PongNoFrameskip-v4.zip"},
]

LOCAL_MODELS = [
    {"label": "PPO  (yours, 10M, Pong)", "algo": "ppo", "path": "models/ppo_10m_Pong_v5/ppo_final.zip"},
    {"label": "DQN  (yours, 10M, Pong)", "algo": "dqn", "path": "models/dqn_10m_Pong_v5/dqn_final.zip"},
    {"label": "A2C  (yours, 10M, Pong)", "algo": "a2c", "path": "models/a2c_10m_Pong_v5/a2c_final.zip"},
]


def make_env(env_id: str):
    env = make_atari_env(env_id, n_envs=1, seed=999)
    return VecFrameStack(env, n_stack=4)


def evaluate(model, env_id: str, n_episodes: int):
    env = make_env(env_id)
    mean, std = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True)
    env.close()
    return mean, std


def load_hf_model(algo: str, repo: str, filename: str):
    from huggingface_sb3 import load_from_hub
    path = load_from_hub(repo_id=repo, filename=filename)
    return ALGO_MAP[algo].load(path)


def print_table(rows: list):
    col_w = [36, 10, 10]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header = "| {:<{}} | {:<{}} | {:<{}} |".format(
        "Model", col_w[0], "Mean", col_w[1], "Std", col_w[2]
    )
    print(sep)
    print(header)
    print(sep)
    for label, mean, std in rows:
        flag = " <-- YOURS" if "yours" in label else ""
        print("| {:<{}} | {:>{}.2f} | {:>{}.2f} |".format(
            label + flag, col_w[0], mean, col_w[1], std, col_w[2]
        ))
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per model")
    parser.add_argument("--env", default="ALE/Pong-v5")
    parser.add_argument("--skip_hf", action="store_true", help="Skip HuggingFace models")
    args = parser.parse_args()

    rows = []

    if not args.skip_hf:
        print("\nDownloading and evaluating HuggingFace pretrained models...")
        print("(requires: pip install huggingface_sb3)\n")
        for m in HF_MODELS:
            print(f"  Loading {m['repo']} ...")
            try:
                model = load_hf_model(m["algo"], m["repo"], m["file"])
                mean, std = evaluate(model, args.env, args.episodes)
                print(f"  {m['label']}: {mean:.2f} +/- {std:.2f}")
                rows.append((m["label"], mean, std))
            except Exception as e:
                print(f"  SKIPPED ({e})")
                rows.append((m["label"], float("nan"), float("nan")))

    print("\nEvaluating your trained models...")
    for m in LOCAL_MODELS:
        print(f"  Loading {m['path']} ...")
        try:
            env = make_env(args.env)
            model = ALGO_MAP[m["algo"]].load(m["path"], env=env)
            mean, std = evaluate_policy(model, env, n_eval_episodes=args.episodes, deterministic=True)
            env.close()
            print(f"  {m['label']}: {mean:.2f} +/- {std:.2f}")
            rows.append((m["label"], mean, std))
        except Exception as e:
            print(f"  SKIPPED ({e})")
            rows.append((m["label"], float("nan"), float("nan")))

    print(f"\n{'='*62}")
    print(f"  RESULTS  ({args.episodes} episodes each, env: {args.env})")
    print(f"{'='*62}")
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
