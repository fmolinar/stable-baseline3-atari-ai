"""Train A2C on an Atari environment with checkpoint saving."""
import os
import argparse
import yaml
import gymnasium as gym
import ale_py
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

gym.register_envs(ale_py)


def train(env_id: str, config: dict, save_dir: str, run_name: str):
    os.makedirs(save_dir, exist_ok=True)
    n_envs = config.get("n_envs", 4)

    vec_env = make_atari_env(env_id, n_envs=n_envs, seed=42)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = make_atari_env(env_id, n_envs=1, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=os.path.join(save_dir, run_name),
        name_prefix="a2c",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, run_name, "best"),
        log_path=os.path.join(save_dir, run_name, "eval_logs"),
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model = A2C(
        "CnnPolicy",
        vec_env,
        learning_rate=config.get("learning_rate", 7e-4),
        n_steps=config.get("n_steps", 5),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 1.0),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.25),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        verbose=1,
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=run_name,
    )
    model.save(os.path.join(save_dir, run_name, "a2c_final"))
    vec_env.close()
    eval_env.close()
    print(f"Training complete. Model saved to {save_dir}/{run_name}/")


def resume(checkpoint_path: str, env_id: str, additional_steps: int, save_dir: str, run_name: str, n_envs: int = 4):
    vec_env = make_atari_env(env_id, n_envs=n_envs, seed=42)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = A2C.load(checkpoint_path, env=vec_env)
    print(f"Resumed from {checkpoint_path} at timestep {model.num_timesteps}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=os.path.join(save_dir, run_name),
        name_prefix="a2c_resumed",
    )
    model.learn(
        total_timesteps=additional_steps,
        callback=checkpoint_cb,
        reset_num_timesteps=False,
    )
    model.save(os.path.join(save_dir, run_name, "a2c_resumed_final"))
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ALE/Pong-v5")
    parser.add_argument("--config", default="configs/a2c_config.yaml")
    parser.add_argument("--experiment", default="a2c_default")
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--resume_steps", type=int, default=500_000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_cfg = next(
        (e for e in cfg["experiments"] if e["name"] == args.experiment),
        cfg["defaults"],
    )

    env_short = args.env.split("/")[-1].replace("-", "_")
    run_name = f"{args.experiment}_{env_short}"

    if args.resume_from:
        resume(args.resume_from, args.env, args.resume_steps, args.save_dir, run_name,
               n_envs=exp_cfg.get("n_envs", 4))
    else:
        train(args.env, exp_cfg, args.save_dir, run_name)
