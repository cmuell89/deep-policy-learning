import torch
import torch.nn as nn
from torch.distributions import Normal

from lr_challenge.learning.policy import GaussianActorPolicy
from lr_challenge.learning.functions import (
    vanilla_advantage,
    normalize_tensors,
)


class VanillaPolicyGradient:
    def __init__(
        self,
        policy: GaussianActorPolicy,
        value_network: nn.Module,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        seed: int = None,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.policy = policy
        self.value_network = value_network
        self.learning_rate = learning_rate
        # Create optimizers with correct parameters
        self.optimizer_policy = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.optimizer_value_network = torch.optim.Adam(
            self.value_network.parameters(), lr=learning_rate
        )

        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        if seed is not None:
            torch.manual_seed(seed)
        self.kwargs = kwargs

        # Get action bounds from policy model - directly use the tensors from the model
        self.action_scale = policy.model.out_scale
        self.action_bias = policy.model.out_bias

        # Initialize log_stds to a smaller value for more precise initial actions
        with torch.no_grad():
            self.policy.log_stds.data.fill_(-2.0)  # exp(-2) ≈ 0.135 standard deviation

    def update(self, observations, actions, rewards, dones):
        observations = observations.to(self.device)
        # normalized_observations = normalize_tensors(observations)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Scale rewards for better learning

        # Get values once
        values = self.value_network(observations).squeeze(-1)

        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones)
        advantages = normalize_tensors(self._compute_advantages(returns, values, dones))

        # Policy update with entropy regularization
        means = self.policy(observations)
        stds = torch.exp(self.policy.log_stds)
        dist = Normal(means, stds)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Add entropy bonus to encourage exploration
        entropy = dist.entropy().mean()
        policy_loss = -(log_probs * advantages).mean() - 0.01 * entropy

        # Perform policy gradient update with clipping
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Value update with clipping
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

    def _compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized discounted returns."""
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
        advantages = vanilla_advantage(returns, values)
        return advantages

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
    ):
        stats = {
            # Returns statistics
            "returns_mean": returns.mean().item(),
            "returns_std": returns.std().item(),
            "returns_min": returns.min().item(),
            "returns_max": returns.max().item(),
            # Loss values
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
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
