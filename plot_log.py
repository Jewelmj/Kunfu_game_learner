import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from scipy.ndimage import uniform_filter1d

from config import *

plt.style.use(PLOT_STYLE)

def smooth_data(data, window):
    """Apply moving average smoothing."""
    if window <= 1 or len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')


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
    ax.scatter(x, y, color=color, s=80, edgecolor="white", linewidth=2, zorder=5)
    ax.annotate(f"{label}\n{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=10,
                color=color,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8, edgecolor=color, linewidth=1.5))


def set_axis_limits(ax, x_min, x_max, y_min, y_max):
    """Set axis limits if specified."""
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)


def configure_plot_aesthetics(ax, title, xlabel, ylabel):
    """Apply consistent aesthetics to plot."""
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    # Format X-axis as plain numbers with commas
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=PLOT_GRID_ALPHA, linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)


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

    n_logs = len(valid_logs)
    colors = PLOT_COLORS[:n_logs] if n_logs <= len(PLOT_COLORS) else plt.cm.tab20(np.linspace(0, 1, n_logs))

    # ===== PLOT 1: Loss vs Steps =====
    fig_loss, ax_loss = plt.subplots(figsize=PLOT_FIGSIZE)
    
    for i, (path, df) in enumerate(valid_logs):
        label = os.path.basename(path).replace('_training_log', '').replace('.csv', '')
        
        steps = df['step'].values
        loss = smooth_data(df['loss'].values, PLOT_SMOOTHING_WINDOW)
        
        ax_loss.plot(steps, loss, label=label, color=colors[i], 
                    linewidth=PLOT_LINEWIDTH, alpha=PLOT_ALPHA)
        annotate_final_point(ax_loss, steps[-1], loss[-1], label, colors[i])
    
    configure_plot_aesthetics(ax_loss, "Training Loss Progression", "Training Steps", "Loss")
    set_axis_limits(ax_loss, PLOT_X_MIN, PLOT_X_MAX, PLOT_LOSS_Y_MIN, PLOT_LOSS_Y_MAX)
    ax_loss.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')
    fig_loss.tight_layout()

    # ===== PLOT 2: Rewards vs Steps =====
    fig_rew, ax_rew = plt.subplots(figsize=PLOT_FIGSIZE)
    
    for i, (path, df) in enumerate(valid_logs):
        label = os.path.basename(path).replace('_training_log', '').replace('.csv', '')
        color = colors[i]
        
        steps = df['step'].values
        avg_reward = smooth_data(df['avg_reward'].values, PLOT_SMOOTHING_WINDOW)
        max_reward = smooth_data(df['max_reward'].values, PLOT_SMOOTHING_WINDOW)
        
        # Plot average reward (solid line)
        ax_rew.plot(steps, avg_reward, label=f"{label} (Avg)", color=color, 
                   linestyle='-', linewidth=PLOT_LINEWIDTH, alpha=PLOT_ALPHA)
        
        # Plot max reward (dashed line, lighter)
        ax_rew.plot(steps, max_reward, label=f"{label} (Max)", color=color, 
                   linestyle='--', linewidth=PLOT_LINEWIDTH - 0.5, alpha=PLOT_ALPHA * 0.7)
        
        # Annotate final points
        last_step = steps[-1]
        last_avg = avg_reward[-1]
        last_max = max_reward[-1]
        
        ax_rew.scatter(last_step, last_avg, color=color, s=80, edgecolor="white", 
                      linewidth=2, zorder=5, marker='o')
        ax_rew.scatter(last_step, last_max, color=color, s=80, edgecolor="white", 
                      linewidth=2, zorder=5, marker='s')
        
        # Smart annotation positioning
        avg_offset = 15 if i % 2 == 0 else -25
        max_offset = -25 if i % 2 == 0 else 15
        
        ax_rew.annotate(f"{last_avg:.1f}",
                       (last_step, last_avg),
                       textcoords="offset points",
                       xytext=(12, avg_offset),
                       fontsize=10,
                       color=color,
                       weight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                alpha=0.9, edgecolor=color, linewidth=1.5))
        
        ax_rew.annotate(f"{last_max:.1f}",
                       (last_step, last_max),
                       textcoords="offset points",
                       xytext=(12, max_offset),
                       fontsize=10,
                       color=color,
                       weight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                alpha=0.9, edgecolor=color, linewidth=1.5))
        
        # Mark peak average reward
        max_idx = np.argmax(avg_reward)
        peak_value = avg_reward[max_idx]
        peak_step = steps[max_idx]
        
        ax_rew.annotate(f"Peak: {peak_value:.1f}",
                       (peak_step, peak_value),
                       textcoords="offset points",
                       xytext=(15, 25),
                       arrowprops=dict(
                           arrowstyle='->', 
                           color=color, 
                           lw=2, 
                           alpha=0.8,
                           connectionstyle="arc3,rad=0.3"
                       ),
                       fontsize=10,
                       color=color,
                       weight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                alpha=0.9, edgecolor=color, linewidth=1.5))
    
    configure_plot_aesthetics(ax_rew, "Training Performance: Average & Maximum Rewards", 
                            "Training Steps", "Reward")
    set_axis_limits(ax_rew, PLOT_X_MIN, PLOT_X_MAX, PLOT_REWARD_Y_MIN, PLOT_REWARD_Y_MAX)
    ax_rew.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='black', ncol=2)
    fig_rew.tight_layout()

    # ===== Save or Show =====
    if save:
        out_path = "results/plots/"
        os.makedirs(out_path, exist_ok=True)
        
        fig_loss.savefig(os.path.join(out_path, "loss_vs_steps.png"), 
                        dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
        fig_rew.savefig(os.path.join(out_path, "reward_vs_steps.png"), 
                       dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
        print(f"✅ Plots saved to {out_path}")
        print(f"   - Resolution: {PLOT_DPI} DPI")
        print(f"   - Smoothing window: {PLOT_SMOOTHING_WINDOW}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs with enhanced styling.")
    parser.add_argument("--save", action="store_true", help="Save plots as PNG instead of showing.")
    args = parser.parse_args()

    if LOG_FILES_TO_PLOT:
        plot_training_logs(LOG_FILES_TO_PLOT, save=args.save)
    else:
        print("❌ No log files specified. Please update LOG_FILES_TO_PLOT in config.py")