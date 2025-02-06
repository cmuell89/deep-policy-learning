
import matplotlib.pyplot as plt
import numpy as np


def save_training_plots(returns, episode_lengths, stats_history, plots_dir):
    # Use a simple style instead of seaborn
    plt.rcParams.update(
        {
            "figure.figsize": [10, 6],
            "figure.dpi": 100,
            "figure.autolayout": True,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2,
        }
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot returns
    ax1.plot(returns, alpha=0.6, label="Episode Return")
    ax1.plot(
        np.convolve(returns, np.ones(100) / 100, mode="valid"),
        label="100-episode Moving Average",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title("Training Returns")
    ax1.legend()
    ax1.grid(True)

    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.6, label="Episode Length")
    ax2.plot(
        np.convolve(episode_lengths, np.ones(100) / 100, mode="valid"),
        label="100-episode Moving Average",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode Lengths")
    ax2.legend()
    ax2.grid(True)

    # Plot losses
    ax3.plot(stats_history["actor_loss"], label="Actor Loss", alpha=0.6)
    ax3.plot(stats_history["critic_loss"], label="Critic Loss", alpha=0.6)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Actor and Critic Losses")
    ax3.legend()
    ax3.grid(True)

    # Plot policy statistics
    ax4.plot(stats_history["action_mean"], label="Action Mean", alpha=0.6)
    ax4.plot(stats_history["policy_std"], label="Policy STD", alpha=0.6)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Value")
    ax4.set_title("Policy Statistics")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/training_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save individual metrics
    for key in stats_history:
        plt.figure(figsize=(10, 5))
        plt.plot(stats_history[key], alpha=0.6)
        plt.title(f"{key} over Episodes")
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.grid(True)
        plt.savefig(f"{plots_dir}/{key}.png", dpi=300, bbox_inches="tight")
        plt.close()