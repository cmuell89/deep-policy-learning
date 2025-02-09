import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Any, DefaultDict

from lr_challenge.learning.policy import GaussianActorPolicy
from lr_challenge.learning.transformations import normalize_tensors
from lr_challenge.learning.functions import (
    vanilla_advantage,
)
from lr_challenge.util import get_output_frequency


class VanillaPolicyGradient:
    """
    Implementation of Vanilla Policy Gradient (VPG) algorithm.

    This class implements the VPG algorithm for continuous action spaces.
    It supports both policy and value function optimization, advantage estimation,
    and includes comprehensive logging capabilities.
    """

    def __init__(
        self,
        policy: GaussianActorPolicy,
        value_network: nn.Module,
        action_dim: int,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        entropy_coef: float = 0.1,
        max_grad_norm: float = 1.0,
        device: str = "cuda:0",
        seed: Optional[int] = None,
        max_steps: int = 1000,
        num_episodes: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize VPG algorithm with explicit parameters.

        Args:
            policy: Policy network
            value_network: Value network
            action_dim: Dimension of action space
            gamma: Discount factor
            learning_rate: Learning rate for both policy and value networks
            entropy_coef: Entropy coefficient for exploration
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ("cpu" or "cuda:0")
            seed: Random seed
            max_steps: Maximum steps per episode
            num_episodes: Number of episodes to train
            **kwargs: Additional arguments
        """
        self.device = torch.device(device)
        self.policy = policy
        self.value_network = value_network
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # Create optimizers with correct parameters
        self.optimizer_policy = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.optimizer_value_network = torch.optim.Adam(
            self.value_network.parameters(), lr=learning_rate
        )

        self.action_dim = action_dim
        self.gamma = gamma
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self.kwargs = kwargs

        # Get action bounds from policy model - directly use the tensors from the model
        self.action_scale = policy.model.out_scale
        self.action_bias = policy.model.out_bias

        # Initialize log_stds to a smaller value for more precise initial actions
        with torch.no_grad():
            self.policy.log_stds.data.fill_(-2.0)  # exp(-2) ≈ 0.135 standard deviation

        # Training configuration
        self.max_steps = max_steps
        self.num_episodes = num_episodes

        # Initialize training state
        self.total_steps = 0
        self.episode_count = 0
        self.writer = None  # Will be set by train()

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of the VPG instance.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters
        """
        return {
            "max_steps": self.max_steps,
            "num_episodes": self.num_episodes,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "action_dim": self.action_dim,
            "device": self.device,
            "seed": self.seed,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
        }

    def _process_trajectory(self, trajectory: Dict[str, List[Any]]) -> Tuple[torch.Tensor, ...]:
        """
        Convert trajectory to tensor batch for update.

        Args:
            trajectory: Dictionary containing trajectory data

        Returns:
            Tuple of tensors containing processed observations, actions, rewards, and dones
        """
        observations = torch.FloatTensor(np.array(trajectory["observations"])).to(
            self.device
        )
        # Actions derived from policy model and are already tensors.
        actions = torch.FloatTensor(trajectory["actions"]).to(self.device)
        rewards = torch.FloatTensor(np.array(trajectory["rewards"])).to(self.device)
        dones = torch.FloatTensor(np.array(trajectory["dones"])).to(self.device)
        return observations, actions, rewards, dones

    def _log_training_stats(self, stats: Dict[str, float], episode: int) -> None:
        """
        Log training metrics to tensorboard.

        Args:
            stats: Dictionary containing training statistics
            episode: Current episode number
        """
        # Critical metrics
        self.writer.add_scalar("1_critical/episode_reward", stats["reward"], episode)
        self.writer.add_scalar("1_critical/policy_loss", stats["policy_loss"], episode)
        self.writer.add_scalar("1_critical/value_loss", stats["value_loss"], episode)

    def _compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized discounted returns.

        Args:
            rewards: Tensor of rewards for each timestep
            dones: Tensor of done flags for each timestep

        Returns:
            Normalized discounted returns
        """
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _compute_advantages(
        self, returns: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages using the vanilla advantage estimation.

        Args:
            returns: Computed returns
            values: Value estimates
            dones: Done flags

        Returns:
            Computed advantages
        """
        advantages = vanilla_advantage(returns, values)
        return advantages

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
        actor_loss: torch.Tensor,
        critic_loss: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute various statistics for logging and monitoring.

        Args:
            actor_loss: Current policy loss value
            critic_loss: Current value function loss
            returns: Computed returns
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
            "returns_mean": returns.mean().item(),
            "returns_std": returns.std().item(),
            "returns_min": returns.min().item(),
            "returns_max": returns.max().item(),
            # Loss values
            "policy_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            # Advantage statistics
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "advantages_min": advantages.min().item(),
            "advantages_max": advantages.max().item(),
            # Action distribution statistics
            "action_mean": means.mean().item(),
            "action_std": means.std().item(),
            "policy_std": stds.mean().item(),  # Current policy standard deviation
            # Value statistics
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
            # Policy statistics
            "log_prob_mean": log_probs.mean().item(),
            "log_prob_std": log_probs.std().item(),
        }
        return stats

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
            f"{YELLOW}Returns μ/σ:{ENDC} {BOLD}{stats['returns_mean']:>8.2f}{ENDC}/{BOLD}{stats['returns_std']:<8.2f}{ENDC} | "
            f"{YELLOW}Policy σ:{ENDC} {BOLD}{stats['policy_std']:>8.3f}{ENDC}\n"
            f"{CYAN}Advantages μ/σ:{ENDC} {BOLD}{stats['advantages_mean']:>8.2f}{ENDC}/{BOLD}{stats['advantages_std']:<8.2f}{ENDC} | "
            f"{CYAN}min/max:{ENDC} {BOLD}{stats['advantages_min']:>8.2f}{ENDC}/{BOLD}{stats['advantages_max']:<8.2f}{ENDC}\n"
            f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{ENDC}\n"
        )

    def rollout(self, env: gym.Env) -> Tuple[Dict[str, float], Dict[str, List[Any]]]:
        """
        Collect single episode of experience from the environment.

        Args:
            env: Gymnasium environment to interact with

        Returns:
            Tuple containing episode information and trajectory data
        """
        obs, _ = env.reset()
        done = 0
        episode_reward = 0

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        while not done and len(trajectory["observations"]) < self.max_steps:
            # Get action
            action, action_info = self.policy.get_action(obs)

            # Step environment
            next_obs, reward, done, truncated, _ = env.step(
                action.detach().cpu().numpy()
            )
            done = 1 if done or truncated else 0
            # Store transition
            trajectory["observations"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(1 if done or truncated else 0)

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

        for episode in range(self.num_episodes):
            # Collect experience
            episode_info, trajectory = self.rollout(env)

            # Convert trajectory to tensors
            batch = self._process_trajectory(trajectory)

            # Update policy and value function
            update_info = self.update(*batch)

            # Combine all info
            train_stats = {**episode_info, **update_info}

            # Log everything
            if self.writer is not None:
                self._log_training_stats(train_stats, episode)

            # Store history
            for k, v in train_stats.items():
                training_info[k].append(v)

            # Evaluate if needed
            eval_stats = self.evaluate(env)
            if self.writer is not None:
                self._log_eval_stats(eval_stats, episode)

            self.episode_count += 1
            # Progress update
            if episode % get_output_frequency(self.num_episodes) == 0:
                self._print_progress(train_stats)

            if episode % get_output_frequency(self.num_episodes) == 0:
                if video_env is not None:
                    video_env.current_episode = episode
                    self._log_video(
                        video_env,
                    )

        return dict(training_info)

    def update(
        self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update policy and value networks using VPG algorithm.

        Args:
            observations: Batch of environment observations
            actions: Batch of actions taken
            rewards: Batch of rewards received
            dones: Batch of done flags

        Returns:
            Dictionary containing update statistics
        """
        observations = observations.to(self.device).detach()
        actions = actions.to(self.device).detach()
        rewards = rewards.to(self.device).detach()
        dones = dones.to(self.device).detach()

        # Get values
        values = self.value_network(observations).squeeze(-1)

        # Compute returns and advantages (returns needs to be non-detached for critic loss)
        returns = self._compute_returns(rewards, dones)
        advantages = normalize_tensors(
            self._compute_advantages(returns, values.detach(), dones)
        )
        # Policy update with entropy regularization
        means = self.policy(observations)
        stds = torch.exp(self.policy.log_stds)
        dist = Normal(means, stds)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        entropy = dist.entropy().mean()
        policy_loss = -(log_probs * advantages).mean() - 0.1 * entropy

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer_policy.step()

        # Value update - ensure returns requires grad
        returns = returns.detach().requires_grad_(True)  # Make returns require grad
        critic_loss = nn.MSELoss()(values, returns)

        self.optimizer_value_network.zero_grad()
        critic_loss.backward()
        self.optimizer_value_network.step()

        # Gradually decrease policy std
        with torch.no_grad():
            self.policy.log_stds.data.clamp_(min=-3.0, max=-1.0)

        return self._compute_stats(
            policy_loss,
            critic_loss,
            rewards,
            values.detach(),
            advantages,
            means.detach(),
            stds,
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
