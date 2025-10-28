import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from config import LOG_FILES_TO_PLOT

plt.style.use('seaborn-v0_8-whitegrid')

def load_log_file(file_path):
    """Load CSV or TSV file safely."""
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
        required_cols = {'step', 'avg_reward', 'max_reward', 'loss'}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Missing columns in {file_path}, skipping.")
            return None
        return df
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return None


def annotate_final_point(ax, x, y, label, color):
    """Add annotation for the final value on the line."""
    ax.scatter(x, y, color=color, s=60, edgecolor="black", zorder=5)
    ax.annotate(f"{label}: {y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=9,
                color=color,
                weight="bold")


def plot_training_logs(log_files, save=False):
    """Plot epsilon, loss, and reward curves from one or more training logs."""
    valid_logs = []
    for f in log_files:
        df = load_log_file(f)
        if df is not None:
            valid_logs.append((f, df))

    if not valid_logs:
        print("❌ No valid log files to plot.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_logs)))

    # Plot 1: Epsilon vs Steps
    fig_eps, ax_eps = plt.subplots(figsize=(12, 6))
    for (i, (path, df)) in enumerate(valid_logs):
        if 'epsilon' not in df.columns:
            continue
        label = os.path.basename(path).replace('_training_log', '').split('.')[0]
        ax_eps.plot(df['step'], df['epsilon'], label=label, color=colors[i], linewidth=2)
        annotate_final_point(ax_eps, df['step'].iloc[-1], df['epsilon'].iloc[-1], label, colors[i])
    ax_eps.set_title("Epsilon Decay Over Training Steps", fontsize=16, pad=10)
    ax_eps.set_xlabel("Steps", fontsize=13)
    ax_eps.set_ylabel("Epsilon", fontsize=13)
    ax_eps.legend()
    ax_eps.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax_eps.grid(True, linestyle="--", alpha=0.6)
    fig_eps.tight_layout()

    # Plot 2: Loss vs Steps
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    for (i, (path, df)) in enumerate(valid_logs):
        label = os.path.basename(path).replace('_training_log', '').split('.')[0]
        ax_loss.plot(df['step'], df['loss'], label=label, color=colors[i], linewidth=2)
        annotate_final_point(ax_loss, df['step'].iloc[-1], df['loss'].iloc[-1], label, colors[i])
    ax_loss.set_title("Training Loss Over Steps", fontsize=16, pad=10)
    ax_loss.set_xlabel("Steps", fontsize=13)
    ax_loss.set_ylabel("Loss", fontsize=13)
    ax_loss.legend()
    ax_loss.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    fig_loss.tight_layout()

    # Plot 3: Rewards vs Steps
    fig_rew, ax_rew = plt.subplots(figsize=(12, 6))
    for (i, (path, df)) in enumerate(valid_logs):
        label = os.path.basename(path).replace('_training_log', '').split('.')[0]
        color = colors[i]

        ax_rew.plot(df['step'], df['avg_reward'], label=f"{label} Avg", color=color, linestyle='-', linewidth=2)
        ax_rew.plot(df['step'], df['max_reward'], label=f"{label} Max", color=color, linestyle='--', linewidth=2)

        last_step = df['step'].iloc[-1]
        last_avg = df['avg_reward'].iloc[-1]
        last_max = df['max_reward'].iloc[-1]

        avg_offset = np.std(df['avg_reward']) * 0.15
        max_offset = -np.std(df['avg_reward']) * 0.15

        ax_rew.scatter(last_step, last_avg, color=color, s=60, edgecolor="black", zorder=5)
        ax_rew.annotate(f"{label} Avg: {last_avg:.1f}",
                        (last_step, last_avg),
                        textcoords="offset points",
                        xytext=(10, avg_offset + 5),
                        fontsize=9,
                        color=color,
                        weight="bold")

        ax_rew.scatter(last_step, last_max, color=color, s=60, edgecolor="black", zorder=5)
        ax_rew.annotate(f"{label} Max: {last_max:.1f}",
                        (last_step, last_max),
                        textcoords="offset points",
                        xytext=(10, max_offset - 5),
                        fontsize=9,
                        color=color,
                        weight="bold")

        max_idx = df['avg_reward'].idxmax()
        if not np.isnan(max_idx):
            ax_rew.annotate(f"Peak {df['avg_reward'][max_idx]:.1f}",
                            (df['step'][max_idx], df['avg_reward'][max_idx]),
                            textcoords="offset points",
                            xytext=(10, 20),
                            arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8, alpha=0.8),
                            fontsize=9,
                            color=color)

    ax_rew.set_title("Average & Max Rewards Over Steps", fontsize=16, pad=10)
    ax_rew.set_xlabel("Steps", fontsize=13)
    ax_rew.set_ylabel("Reward", fontsize=13)
    ax_rew.legend()
    ax_rew.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax_rew.grid(True, linestyle="--", alpha=0.6)
    fig_rew.tight_layout()

    if save:
        out_path = "results/plots/"
        os.makedirs(out_path, exist_ok=True)
        fig_eps.savefig(os.path.join(out_path, "epsilon_vs_steps.png"), dpi=300, bbox_inches='tight')
        fig_loss.savefig(os.path.join(out_path, "loss_vs_steps.png"), dpi=300, bbox_inches='tight')
        fig_rew.savefig(os.path.join(out_path, "reward_vs_steps.png"), dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs with nice annotations.")
    parser.add_argument("--save", action="store_true", help="Save plots as PNG instead of showing.")
    args = parser.parse_args()

    if LOG_FILES_TO_PLOT:
        plot_training_logs(LOG_FILES_TO_PLOT, save=args.save)
    else:
        print("❌ No log files specified. Please update LOG_FILES_TO_PLOT.")
