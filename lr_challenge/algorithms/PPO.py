import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional, Dict, List, Any, DefaultDict

from lr_challenge.learning.policy import GaussianActorPolicy
from lr_challenge.learning.transformations import (
    normalize_tensors,
)
from lr_challenge.learning.functions import (
    generalized_advantage_estimate,
    compute_surrogate_loss,
)
from lr_challenge.util import get_output_frequency


class ProximalPolicyOptimization:
    """
    Implementation of Proximal Policy Optimization (PPO) algorithm.

    This class implements the PPO algorithm with clipped objective function for continuous action spaces.
    It supports both policy and value function optimization, Generalized Advantage Estimation (GAE),
    and includes comprehensive logging capabilities.
    """

    def __init__(
        self,
        policy: GaussianActorPolicy,
        value_network: nn.Module,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        learning_rate: float = 3e-4,
        max_steps: int = 1000,
        num_episodes: int = 100,
        trajectories_per_episode: int = 4,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """Initialize PPO algorithm with explicit parameters.

        Args:
            policy: Policy network
            value_network: Value network
            action_dim: Dimension of action space
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clipping_epsilon: PPO clipping parameter
            n_epochs: Number of epochs per update
            batch_size: Mini-batch size for updates
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            learning_rate: Learning rate
            max_steps: Maximum steps per episode
            num_episodes: Number of episodes to train
            trajectories_per_episode: Number of trajectories to collect before update
            device: Device to run on ("cpu" or "cuda")
            seed: Random seed
        """
        self.device = torch.device(device)
        self.policy = policy
        self.value_network = value_network
        self.action_dim = action_dim

        # PPO specific parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clipping_epsilon = clipping_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate

        # Initialize optimizers
        self.optimizer_policy = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )
        self.optimizer_value = torch.optim.Adam(
            self.value_network.parameters(), lr=self.learning_rate
        )

        # Training configuration
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.trajectories_per_episode = trajectories_per_episode

        # Initialize training state
        self.total_steps = 0
        self.episode_count = 0
        self.writer = None
        self.seed = seed
        # Set seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Get action bounds from policy model
        self.action_scale = policy.model.in_scale
        self.action_bias = policy.model.in_bias

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of the PPO instance.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters
        """
        return {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clipping_epsilon": self.clipping_epsilon,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "num_episodes": self.num_episodes,
            "trajectories_per_episode": self.trajectories_per_episode,
            "seed": self.seed,
            "device": self.device,
        }

    def _log_training_stats(self, stats: Dict[str, float], episode: int) -> None:
        """
        Log training metrics to tensorboard.

        Args:
            stats: Dictionary containing training statistics
            episode: Current episode number
        """
        self.writer.add_scalar("1_critical/episode_reward", stats["reward"], episode)
        self.writer.add_scalar("1_critical/policy_loss", stats["policy_loss"], episode)
        self.writer.add_scalar("1_critical/value_loss", stats["value_loss"], episode)

    def _log_eval_stats(self, stats: Dict[str, float], episode: int) -> None:
        """
        Log evaluation metrics to tensorboard.

        Args:
            stats: Dictionary containing evaluation statistics
            episode: Current episode number
        """
        self.writer.add_scalar("2_eval/mean_reward", stats["eval_reward_mean"], episode)
        self.writer.add_scalar("2_eval/std_reward", stats["eval_reward_std"], episode)
        self.writer.add_scalar(
            "2_eval/mean_episode_length", stats["eval_length_mean"], episode
        )

    def _print_progress(self, stats: Dict[str, float]) -> None:
        """
        Print formatted training progress to console with colors.

        Args:
            stats: Dictionary containing training statistics to display
        """
        # ANSI color codes
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        CYAN = "\033[96m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"

        print(
            f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{ENDC}\n"
            f"{BLUE}Episode:{ENDC} {BOLD}{self.episode_count}{ENDC} | "
            f"{BLUE}Steps:{ENDC} {BOLD}{self.total_steps:,}{ENDC}\n"
            f"{GREEN}Reward:{ENDC} {BOLD}{stats['reward']:>8.2f}{ENDC} | "
            f"{RED}Policy Loss:{ENDC} {BOLD}{stats['policy_loss']:>8.3f}{ENDC} | "
            f"{RED}Value Loss:{ENDC} {BOLD}{stats['value_loss']:>8.3f}{ENDC}\n"
            f"{YELLOW}Policy σ:{ENDC} {BOLD}{stats['policy_std']:>8.3f}{ENDC}\n"
            f"{CYAN}Advantages μ/σ:{ENDC} {BOLD}{stats['advantages_mean']:>8.2f}{ENDC}/{BOLD}{stats['advantages_std']:<8.2f}{ENDC} | "
            f"{CYAN}min/max:{ENDC} {BOLD}{stats['advantages_min']:>8.2f}{ENDC}/{BOLD}{stats['advantages_max']:<8.2f}{ENDC}\n"
            f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{ENDC}\n"
        )

    def _compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of rewards for each timestep
            values: Tensor of value estimates for each timestep
            dones: Tensor of done flags for each timestep

        Returns:
            Tuple containing normalized returns and advantages
        """
        returns, advantages = generalized_advantage_estimate(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        returns = normalize_tensors(returns)
        advantages = normalize_tensors(advantages)
        return returns, advantages

    def _entropy_loss(self, log_stds: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy loss for a Gaussian distribution using log standard deviations.

        Args:
            log_stds: Log standard deviations of the Gaussian distribution

        Returns:
            entropy_loss: The entropy loss term (negative entropy scaled by coefficient)
        """
        entropy = 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e)) + log_stds
        entropy_loss = -self.ent_coef * entropy.mean()
        return entropy_loss

    def _surrogate_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO surrogate loss with clipping.

        Args:
            advantages: Advantage estimates
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities

        Returns:
            Surrogate loss value
        """
        # Calculate policy ratio and surrogate loss
        surrogate_loss = compute_surrogate_loss(
            advantages, log_probs, old_log_probs, self.clipping_epsilon
        )
        return surrogate_loss

    def _value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clipped value function loss.

        Args:
            values: Current value estimates
            returns: Computed returns
            old_values: Previous value estimates

        Returns:
            Clipped value function loss
        """
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clipping_epsilon,
            self.clipping_epsilon,
        )
        value_loss1 = (values - returns).pow(2)
        value_loss2 = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        return value_loss

    def _log_video(self, video_env: gym.Env) -> None:
        """
        Log video of evaluation episode.

        Args:
            video_env: Gymnasium environment with video recording capability
        """
        with torch.no_grad():
            for n in range(5):
                _, _ = self.rollout(video_env)

    def _compute_stats(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute various statistics for logging and monitoring.

        Args:
            policy_loss: Current policy loss value
            value_loss: Current value function loss
            rewards: Episode rewards
            values: Value estimates
            advantages: Computed advantages
            means: Action distribution means
            stds: Action distribution standard deviations
            log_probs: Log probabilities of actions

        Returns:
            Dictionary containing computed statistics
        """
        stats = {
            # Returns statistics
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "rewards_min": rewards.min().item(),
            "rewards_max": rewards.max().item(),
            # Loss values
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            # Advantage statistics
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "advantages_min": advantages.min().item(),
            "advantages_max": advantages.max().item(),
            # Action distribution statistics
            "action_mean": means.mean().item(),
            "action_std": means.std().item(),
            "policy_std": torch.exp(stds).mean().item(),
            # Value statistics
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
            # Policy statistics
            "log_prob_mean": log_probs.mean().item(),
            "log_prob_std": log_probs.std().item(),
        }
        return stats

    def rollout(self, env: gym.Env) -> Tuple[Dict[str, float], Dict[str, List[Any]]]:
        """
        Collect a single episode of experience from the environment.

        Args:
            env: Gymnasium environment to interact with

        Returns:
            Tuple containing episode information and trajectory data
        """
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        while not done and len(trajectory["observations"]) < self.max_steps:
            if isinstance(obs, dict):
                obs = obs.get("observation", obs)
            action, action_info = self.policy.get_action(obs)

            next_obs, reward, done, truncated, _ = env.step(
                action.detach().cpu().numpy()
            )
            done = 1 if done or truncated else 0

            trajectory["observations"].append(obs)
            trajectory["actions"].append(action.detach().cpu().numpy())
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

            obs = next_obs
            episode_reward += reward
            self.total_steps += 1

        episode_info = {
            "reward": episode_reward,
            "length": len(trajectory),
            "total_steps": self.total_steps,
        }

        return episode_info, trajectory

    def train(
        self,
        env: gym.Env,
        writer: Optional[SummaryWriter] = None,
        video_env: Optional[gym.Env] = None,
    ) -> Dict[str, List[float]]:
        """
        Execute the full training loop with logging.

        Args:
            env: Training environment
            writer: TensorBoard summary writer for logging
            video_env: Optional environment for video recording

        Returns:
            Dictionary containing training history
        """
        self.writer = writer
        training_info = defaultdict(list)

        # Number of rollouts to collect before updating
        rollouts_per_update = (
            self.trajectories_per_episode
        )  # Collect 4 episodes worth of data before updating

        for episode in range(0, self.num_episodes):
            # Storage for multiple rollouts
            all_observations = []
            all_actions = []
            all_rewards = []
            all_dones = []
            episode_rewards = []
            total_steps = 0

            # Collect multiple rollouts
            for _ in range(rollouts_per_update):
                # Collect experience
                episode_info, trajectory = self.rollout(env)

                # Store trajectory data
                all_observations.extend(trajectory["observations"])
                all_actions.extend(trajectory["actions"])
                all_rewards.extend(trajectory["rewards"])
                all_dones.extend(trajectory["dones"])

                # Track episode stats
                episode_rewards.append(episode_info["reward"])
                total_steps += episode_info["length"]

            self.episode_count += 1

            # Convert collected trajectories to tensors
            batch = (
                torch.FloatTensor(np.array(all_observations)).to(self.device),
                torch.FloatTensor(np.array(all_actions)).to(self.device),
                torch.FloatTensor(np.array(all_rewards)).to(self.device),
                torch.FloatTensor(np.array(all_dones)).to(self.device),
            )

            # Update policy and value function using PPO
            update_info = self.update(*batch)

            # Compute average episode statistics
            train_stats = {
                **update_info,
                "reward": np.mean(episode_rewards),
                "reward_std": np.std(episode_rewards),
                "length": total_steps / rollouts_per_update,
                "total_steps": self.total_steps,
            }

            # Log everything
            if self.writer is not None:
                self._log_training_stats(train_stats, episode)

            # Store history
            for k, v in train_stats.items():
                training_info[k].append(v)

            # Progress update
            if episode % get_output_frequency(self.num_episodes) == 0:
                self._print_progress(train_stats)

            if episode % get_output_frequency(self.num_episodes) == 0:
                if video_env is not None:
                    video_env.current_episode = episode
                    self._log_video(
                        video_env,
                    )

            eval_stats = self.evaluate(env)
            if self.writer is not None:
                self._log_eval_stats(eval_stats, episode)

        return dict(training_info)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO algorithm.

        Args:
            observations: Batch of environment observations
            actions: Batch of actions taken
            rewards: Batch of rewards received
            dones: Batch of done flags

        Returns:
            Dictionary containing update statistics
        """
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # return old log probabilities as this establishes the proximal trust region
        _, _, old_log_probs = self.policy.old_stats(observations, actions)
        old_values = self.value_network(observations).squeeze(-1)
        normalized_returns, normalized_advantages = self._compute_advantages(
            rewards, old_values, dones
        )

        batch_size = observations.shape[0]

        for _ in range(self.n_epochs):
            indices = torch.randperm(
                batch_size, device=self.device
            )  # Keep indices on same device

            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                mb_indices = indices[start:end]
                # Get minibatch data
                observations_mb = observations[mb_indices].detach()
                actions_mb = actions[mb_indices].detach()
                advantages_mb = normalized_advantages[mb_indices].detach()
                returns_mb = normalized_returns[mb_indices].detach()
                old_values_mb = old_values[mb_indices].detach()
                old_log_probs_mb = old_log_probs[mb_indices].detach()

                # Get current policy distribution
                _, log_stds_mb, log_probs_mb = self.policy.stats(
                    observations_mb, actions_mb
                )

                # Update policy
                policy_loss_mb = self._surrogate_loss(
                    advantages_mb,
                    log_probs_mb,
                    old_log_probs_mb,
                )
                entropy_loss = self._entropy_loss(log_stds_mb)
                total_policy_loss = policy_loss_mb + entropy_loss

                self.optimizer_policy.zero_grad()
                total_policy_loss.backward()
                self.optimizer_policy.step()

                # Update value network
                values_mb = self.value_network(observations_mb.clone()).squeeze(-1)
                value_loss = self._value_loss(values_mb, returns_mb, old_values_mb)
                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()

        # Get final stats for logging
        mean, log_stds, log_probs = self.policy.stats(observations, actions)
        self.policy.update_old_stats()

        return self._compute_stats(
            policy_loss_mb,
            value_loss,
            rewards,
            values_mb.detach(),
            normalized_advantages,
            mean.detach(),
            log_stds.detach(),
            log_probs.detach(),
        )

    def evaluate(self, env: gym.Env, n_episodes: int = 5) -> Dict[str, float]:
        """
        Run evaluation episodes.

        Args:
            env: Environment to evaluate in
            n_episodes: Number of evaluation episodes to run

        Returns:
            Dictionary containing evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(n_episodes):
            with torch.no_grad():
                episode_info, _ = self.rollout(env)
            eval_rewards.append(episode_info["reward"])
            eval_lengths.append(episode_info["length"])

        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
        }
