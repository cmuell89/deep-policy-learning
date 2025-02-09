import matplotlib.pyplot as plt
import numpy as np
import json
import gymnasium as gym
import os
import cv2
from typing import Optional, Dict, Any


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_output_frequency(count):
    if count < 50:
        return 5
    elif count < 100:
        return 10
    elif count < 1000:
        return count // 50
    else:
        return count // 100


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


class ContinuousToDiscreteWrapper(gym.Wrapper):
    """Wrapper to convert continuous actions to discrete for CartPole."""

    def __init__(self, env):
        super().__init__(env)
        # Override action space to be continuous
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = env.observation_space

    def step(self, action):
        # Convert continuous action to discrete
        # If action > 0, push right (1), else push left (0)
        discrete_action = 1 if action[0] > 0 else 0
        return self.env.step(discrete_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VideoRecorderWrapper(gym.Wrapper):
    """Wrapper for recording videos of environment episodes."""

    def __init__(
        self,
        env: gym.Env,
        video_dir: str,
        video_prefix: str = "episode",
        fps: int = 60,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the video recorder wrapper."""
        super().__init__(env)
        self.video_dir = video_dir
        self.video_prefix = video_prefix
        self.fps = fps
        self.render_kwargs = render_kwargs or {}
        self.current_video = None
        self.current_episode = 0
        self.trial_count = 0

        # Create video directory if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)

        # Reset environment first to get frame dimensions
        self.env.reset()
        frame = self.env.render()
        self.frame_height, self.frame_width = frame.shape[:2]

        # Initialize video writer as None
        self.current_video = None

    def reset(self, **kwargs):
        """Reset the environment and start new video if needed."""
        obs, info = self.env.reset(**kwargs)

        # Close previous video if exists
        if self.current_video is not None:
            self.current_video.release()

        # Create new video file
        video_path = os.path.join(
            self.video_dir, f"{self.video_prefix}_{self.current_episode}_{self.trial_count}.mp4"
        )
        self.current_video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        # Record first frame
        frame = self.env.render()
        self.current_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        return obs, info

    def step(self, action):
        """Step the environment and record frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Record frame
        if self.current_video is not None:  # Check if video writer exists
            frame = self.env.render()
            self.current_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # If episode is done, close video and increment counter
        if terminated or truncated:
            if self.current_video is not None:
                self.current_video.release()
                self.current_video = None
            self.trial_count += 1

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up video writer."""
        if self.current_video is not None:
            self.current_video.release()
        self.env.close()
