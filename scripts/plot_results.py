"""Load eval JSONs and produce learning curve plots + results tables for all environments."""
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EVAL_FILES = {
    "Space Invaders": {
        "DQN": "results/dqn_si_eval.json",
        "A2C": "results/a2c_si_eval.json",
        "PPO": "results/ppo_si_eval.json",
    },
    "Pong": {
        "DQN": "results/dqn_pong_eval.json",
        "A2C": "results/a2c_pong_eval.json",
        "PPO": "results/ppo_pong_eval.json",
    },
}

COLORS = {"DQN": "#1f77b4", "A2C": "#ff7f0e", "PPO": "#2ca02c"}
FINAL_TIMESTEPS = 10_000_000


def parse_timestep(checkpoint_path: str) -> int:
    name = os.path.basename(checkpoint_path)
    if "final" in name:
        return FINAL_TIMESTEPS
    m = re.search(r"_(\d+)_steps", name)
    return int(m.group(1)) if m else 0


def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    rows = sorted([
        {"timestep": parse_timestep(e["checkpoint"]),
         "mean": e["mean_reward"],
         "std": e["std_reward"]}
        for e in data
    ], key=lambda r: r["timestep"])
    # deduplicate final vs 10M (keep whichever comes last)
    seen = {}
    for r in rows:
        seen[r["timestep"]] = r
    return sorted(seen.values(), key=lambda r: r["timestep"])


def fmt_ts(x):
    if x >= 1_000_000:
        return f"{int(x/1e6)}M"
    return f"{int(x/1e3)}k"


def main():
    os.makedirs("report", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (env_name, algo_files) in zip(axes, EVAL_FILES.items()):
        env_data = {}
        for algo, path in algo_files.items():
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                continue
            env_data[algo] = load_results(path)

        for algo, rows in env_data.items():
            xs = [r["timestep"] for r in rows]
            ys = [r["mean"] for r in rows]
            errs = [r["std"] for r in rows]
            ax.plot(xs, ys, marker="o", label=algo, color=COLORS[algo])
            ax.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            alpha=0.15, color=COLORS[algo])

        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel("Mean Episode Reward (20 episodes)")
        ax.set_title(f"DQN vs A2C vs PPO — {env_name} (10M steps)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_ts(x)))
        ax.legend()
        ax.grid(True, alpha=0.3)
        if env_name == "Pong":
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="break-even")

    fig.tight_layout()
    out = "report/learning_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot → {out}\n")

    # --- Markdown results tables ---
    for env_name, algo_files in EVAL_FILES.items():
        print(f"## {env_name} Results\n")
        print("| Algorithm | Timesteps | Mean Reward | Std |")
        print("|---|---|---|---|")
        for algo, path in algo_files.items():
            if not os.path.exists(path):
                continue
            for r in load_results(path):
                print(f"| {algo} | {fmt_ts(r['timestep'])} | {r['mean']:.1f} | {r['std']:.1f} |")
        print()


if __name__ == "__main__":
    main()
