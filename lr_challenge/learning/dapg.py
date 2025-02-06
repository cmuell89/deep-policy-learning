import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Dict, List, Optional
import numpy as np
import os
import copy
from abc import ABC, abstractmethod
from lr_challenge.learning.functions import (
    kl_divergence,
    generalized_advantage_estimate,
    conjugate_gradient,
    gen_hvp_evaluator,
    compute_surrogate_loss,
    compute_policy_gradient,
)


class BasePolicyGradient(ABC):
    """Base class for policy gradient algorithms.
    Assumptions:
      Policy and Value Networks
      GAE advatanges
      Normalized observations
      Value network is updated with returns
      Policy network is updated with natural gradient
    """

    def __init__(
        self,
        policy_network: nn.Module,
        value_network: nn.Module,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
        seed: int = None,
        **kwargs,
    ):
        """
        Args:
            policy_network: Neural network for the policy
            value_network: Neural network for value function
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to run computations on
            seed: Optional seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        self.policy = policy_network.to(device)
        self.log_std = None
        self.old_policy = None
        self.old_log_std = None
        self.value_network = value_network.to(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.action_dim = action_dim

        # Statistics tracking
        self.training_stats = {}
        self.episode_count = 0

        # Running normalization stats
        self.obs_mean = None
        self.obs_std = None

    @abstractmethod
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update policy and value function using collected trajectories"""
        pass

    @abstractmethod
    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute advantages using GAE or other methods

        Args:
            rewards: Rewards tensor [T]
            values: Value estimates [T+1]
            dones: Done flags [T]

        Returns:
            advantages: Computed advantages [T]
        """
        pass

    def get_action(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Get action from policy with additional info"""
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.device)
            action, info = self.policy(obs)
            return action.cpu().numpy(), {
                k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in info.items()
            }

    def mean_log_likelihood(self, actions, mean, log_std):
        """Compute mean and log likelihood of actions under policy

        Args:
            actions: Actions in post-tanh space [-1,1]
            mean: Mean in pre-tanh space
            log_std: Log standard deviation in pre-tanh space

        Returns:
            mean: Policy mean
            log_likelihood: Log likelihood of the actions
        """
        # Transform actions back to pre-tanh space
        normalized_actions = (
            actions - self.policy.action_bias
        ) / self.policy.action_scale
        x_t = torch.atanh(torch.clamp(normalized_actions, -0.999999, 0.999999))

        # Compute log likelihood in pre-tanh space
        zs = (x_t - mean) / torch.exp(log_std)
        log_likelihood = (
            -0.5 * torch.sum(zs**2, dim=1)
            - torch.sum(log_std)
            - 0.5 * self.action_dim * np.log(2 * np.pi)
        )

        # Add log det jacobian for tanh transform
        log_det_jacobian = torch.sum(
            torch.log(torch.clamp(1 - torch.tanh(x_t).pow(2), min=1e-6)), dim=-1
        )
        log_likelihood = log_likelihood - log_det_jacobian

        return mean, log_likelihood

    def get_old_policy_dist(self, observations: torch.Tensor, actions: torch.Tensor):
        """Get distribution info from old policy

        Args:
            observations: Input states/observations tensor

        Returns:
            Tuple of:
                mean: Mean of the policy distribution
                std: Standard deviation (exp(log_std))
                log_std: Log standard deviation
                dist: Normal distribution object
                log_likelihood: Log likelihood of the distribution
        """
        if self.old_policy is None:
            self.old_policy = copy.deepcopy(self.policy)
            _, _, curr_log_std, _, _ = self.get_current_policy_dist(
                observations, actions
            )
            self.old_log_std = curr_log_std
        if self.old_log_std is None:
            _, _, curr_log_std, _, _ = self.get_current_policy_dist(
                observations, actions
            )
            self.old_log_std = curr_log_std

        # Get mean from policy
        _, info = self.old_policy(observations)
        mean = info["mean"]

        mean, log_likelihood = self.mean_log_likelihood(
            actions,
            mean=mean,
            log_std=self.old_log_std,
        )

        return (
            mean,
            torch.exp(self.old_log_std),
            self.old_log_std,
            Normal(mean, torch.exp(self.old_log_std)),
            log_likelihood,
        )

    def get_current_policy_dist(
        self, observations: torch.Tensor, actions: torch.Tensor
    ):
        """Get distribution info from current policy

        Args:
            observations: Input states/observations tensor

        Returns:
            Tuple of:
                mean: Mean of the policy distribution
                std: Standard deviation (exp(log_std))
                log_std: Log standard deviation
                dist: Normal distribution object
                log_likelihood: Log likelihood of the distribution
        """
        # Get mean from policy
        _, info = self.policy(observations)
        mean = info["mean"]
        log_std = info["log_std"]

        mean, log_likelihood = self.mean_log_likelihood(
            actions,
            mean=mean,
            log_std=log_std,
        )
        return (
            mean,
            torch.exp(log_std),
            log_std,
            Normal(mean, torch.exp(log_std)),
            log_likelihood,
        )

    def set_old_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict(), strict=True)
        self.old_log_std = self.log_std

    def get_param_values(self) -> torch.Tensor:
        """Get flattened parameters from policy network"""
        params = []
        for param in self.policy.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def update_value_network(
        self, states: torch.Tensor, returns: torch.Tensor
    ) -> float:
        """Update value network using Adam optimizer

        Args:
            states: Input states
            returns: Target returns

        Returns:
            Value loss (float)
        """
        criterion = nn.MSELoss()

        # Zero gradients
        self.value_optimizer.zero_grad()

        # CHARGE!
        predicted_values = self.value_network(states).squeeze()
        value_loss = criterion(predicted_values, returns)

        # RETREAT!
        value_loss.backward()

        # Clip gradients for stability!
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)

        # Optimizer step
        self.value_optimizer.step()

        return value_loss.item()

    def update_policy_network(
        self, observations: torch.Tensor, new_params: torch.Tensor
    ) -> None:
        """Update policy network parameters with new values

        Args:
            observations: Current batch of observations for computing distributions
            new_params: New flattened parameter vector from natural gradient update
        """
        # Convert flattened parameters back to original shapes
        idx = 0
        for param in self.policy.parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()

            # Extract and reshape parameter segment
            new_param = new_params[idx : idx + param_size].reshape(param_shape)

            # Update parameter
            param.data.copy_(new_param.to(param.device))
            idx += param_size

        # Verify parameters were updated correctly
        if idx != new_params.numel():
            raise ValueError(
                f"Parameter size mismatch. Expected {new_params.numel()}, used {idx}"
            )

    def normalize_tensors(
        self, tensors: torch.Tensor, update_stats: bool = True
    ) -> torch.Tensor:
        """Normalize tensors using self means and stds"""
        if update_stats or self.obs_mean is None:
            mean = tensors.mean(0)
            std = tensors.std(0) + 1e-8

        return (tensors - mean) / std

    def get_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        return self.training_stats

    def save(self, path: str):
        """Save policy and value networks"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value_network.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_std": self.obs_std,
                "training_stats": self.training_stats,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load(self, path: str):
        """Load policy and value networks"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_network.load_state_dict(checkpoint["value_state_dict"])
        self.obs_mean = checkpoint["obs_mean"]
        self.obs_std = checkpoint["obs_std"]
        self.training_stats = checkpoint["training_stats"]
        self.episode_count = checkpoint["episode_count"]

    def process_trajectories(
        self, trajectories: List[Dict]
    ) -> Tuple[torch.Tensor, ...]:
        """Process trajectories into tensors for training

        Args:
            trajectories: List of trajectory dictionaries containing numpy arrays

        Returns:
            Tuple of (observations, actions, rewards, dones) tensors on correct device
        """
        # Create tensors directly on target device
        observations = torch.cat(
            [
                torch.tensor(t["observations"], dtype=torch.float32, device=self.device)
                for t in trajectories
            ]
        )
        actions = torch.cat(
            [
                torch.tensor(t["actions"], dtype=torch.float32, device=self.device)
                for t in trajectories
            ]
        )
        rewards = torch.cat(
            [
                torch.tensor(t["rewards"], dtype=torch.float32, device=self.device)
                for t in trajectories
            ]
        )
        dones = torch.cat(
            [
                torch.tensor(t["dones"], dtype=torch.float32, device=self.device)
                for t in trajectories
            ]
        )

        return observations, actions, rewards, dones

    def log_stats(self, stats: Dict[str, float]):
        """Log training statistics"""
        # Update internal stats
        self.training_stats.update(stats)

        # Print key metrics
        print("\nTraining Stats:")
        print(f"Episode: {self.episode_count}")
        print(f"Mean Return: {stats.get('mean_return', 'N/A'):.3f}")
        print(f"Value Loss: {stats.get('value_loss', 'N/A'):.3f}")
        print(f"Policy Loss: {stats.get('policy_loss', 'N/A'):.3f}")

        self.episode_count += 1


class DAPG(BasePolicyGradient):
    def __init__(
        self,
        policy_network: nn.Module,
        value_network: nn.Module,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
        normalized_step_size: float = 0.01,
        FIM_invert_args: dict = {"iters": 10, "damping": 1e-4},
        hvp_sample_frac: float = 1.0,
        kl_dist: Optional[float] = None,
        lam_0: float = 1.0,
        lam_1: float = 0.95,
        residual_tol: float = 1e-10,
        input_normalization: float = 0.9,
        seed: int = None,
        **kwargs,
    ):
        """Demonstration Augmented Policy Gradient (DAPG) implementation.

        DAPG combines natural policy gradient with demonstration data to accelerate learning.
        Uses GAE for advantage estimation and trust region optimization with KL constraints.

        Args:
            policy_network: Neural network for the policy
            value_network: Neural network for value function estimation
            action_dim: Dimension of the action space
            gamma: Discount factor for rewards (default: 0.99)
            gae_lambda: Lambda parameter for GAE advantage estimation (default: 0.95)
            device: Device to run computations on (default: "cpu")
            normalized_step_size: Step size for trust region constraint (default: 0.01)
            FIM_invert_args: Arguments for conjugate gradient optimization
                - iters: Number of CG iterations
                - damping: Damping coefficient for FIM
            hvp_sample_frac: Fraction of data to use for HVP computation (default: 1.0)
            kl_dist: KL divergence constraint threshold (default: None)
            lam_0: Initial weight for demonstration advantages (default: 1.0)
            lam_1: Decay rate for demonstration advantages (default: 0.95)
            residual_tol: Tolerance for conjugate gradient convergence (default: 1e-10)
            input_normalization: Running average factor for observation normalization (default: 0.9)
            seed: Optional seed for reproducibility
        """
        super().__init__(
            policy_network, value_network, action_dim, gamma, gae_lambda, device, seed
        )
        self.value_learning_rate = 1e-3
        self.delta = kl_dist if kl_dist is not None else 0.5 * normalized_step_size
        self.damping_coeff = FIM_invert_args.get("damping", 0.1)
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0
        self.hvp_subsample = hvp_sample_frac
        self.input_normalization = input_normalization  # Default value
        self.FIM_invert_args = FIM_invert_args
        self.hvp_sample_frac = hvp_sample_frac
        self.kl_dist = kl_dist
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.residual_tol = residual_tol
        self.input_normalization = input_normalization

        # Only use optimizer for value network since policy uses natural gradient
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=self.value_learning_rate
        )

    def demonstration_advantage(self, demo_obs: torch.Tensor) -> torch.Tensor:
        """Compute advantages for demonstration paths"""
        demo_adv = (
            self.lam_0 * (self.lam_1**self.iter_count) * np.ones(demo_obs.shape[0])
        )
        return demo_adv

    def compute_kl_divergence(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between old and new policy distributions"""
        old_means, _, old_log_stds, _, _ = self.get_old_policy_dist(states, actions)
        new_means, _, new_log_stds, _, _ = self.get_current_policy_dist(states, actions)
        print(f"old_means: {old_means}")
        print(f"old_log_stds: {old_log_stds}")
        print(f"new_means: {new_means}")
        print(f"new_log_stds: {new_log_stds}")
        return kl_divergence(old_means, old_log_stds, new_means, new_log_stds)

    def vanilla_policy_gradient(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vanilla policy gradient

        Args:
            observations: States from trajectories
            actions: Actions taken in states
            advantages: Advantage estimates

        Returns:
            Flattened vanilla policy gradient
        """
        _, _, _, curr_dist, _ = self.get_current_policy_dist(observations, actions)
        loss = compute_surrogate_loss(
            curr_dist=curr_dist,
            old_dist=None,
            actions=actions,
            advantages=advantages,
            action_scale=self.policy.action_scale,
            action_bias=self.policy.action_bias,
        )

        # Compute gradient
        vpg_grad = compute_policy_gradient(loss, self.policy)

        return vpg_grad

    def update(
        self, paths: List[Dict], demo_paths: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """Update policy following the DAPG algorithm structure"""
        use_dapg = demo_paths is not None and self.lam_0 > 0.0

        observations, actions, rewards, dones = self.process_trajectories(paths)

        # Normalize observations
        normalized_obs = self.normalize_tensors(observations)

        # Get values for current and next states
        values = self.value_network(normalized_obs).squeeze()
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]  # Last value stays 0 for terminal states

        # Compute advantages BEFORE capturing old distribution
        policy_adv = self.compute_advantages(rewards, values, dones)
        # Handle demonstration paths with proper scaling
        if use_dapg:
            # let's assume no known rewards or dones for demo paths.
            # Demo paths should have tensors for rewards and dones but no values.
            demo_obs, demo_act, _, _ = self.process_trajectories(demo_paths)
            demo_adv = self.demonstration_advantage(demo_obs)
            self.iter_count += 1

            # Normalize demo observations too
            normalized_demo_obs = self.normalize_tensors(demo_obs)

            # Concatenate all data with proper scaling
            combined_obs = torch.cat([normalized_obs, normalized_demo_obs])
            combined_act = torch.cat([actions, demo_act])
            combined_adv = torch.cat([policy_adv, demo_adv])
        else:
            combined_obs = normalized_obs
            combined_act = actions
            combined_adv = policy_adv

        # Get old and current policy distribution
        old_mean, old_std, old_log_std, old_dist, _ = self.get_old_policy_dist(normalized_obs, actions)
        curr_mean, curr_std, curr_log_std, curr_dist, _ = self.get_current_policy_dist(
            normalized_obs, actions
        )
        if self.old_log_std is None:
            self.old_log_std = curr_log_std
            old_dist = Normal(old_mean, torch.exp(self.old_log_std))
        # Compute surrogate loss
        loss = compute_surrogate_loss(
            curr_dist=curr_dist,
            old_dist=old_dist,
            actions=actions,
            advantages=policy_adv,
            action_scale=self.policy.action_scale,
            action_bias=self.policy.action_bias,
        )
        surr_before = loss.item()
        if use_dapg:
            # Compute DAPG gradient with proper scaling. This is the only calculation that uses the demonstrations.
            vanilla_grad_with_demos = self.vanilla_policy_gradient(
                observations=combined_obs,
                actions=combined_act,
                advantages=combined_adv,
            )
            sample_coef = combined_adv.shape[0] / policy_adv.shape[0]
            policy_grad = sample_coef * vanilla_grad_with_demos

        else:
            # Compute regular policy gradient
            policy_grad = self.vanilla_policy_gradient(
                observations=normalized_obs,
                actions=actions,
                advantages=policy_adv,
            )

        # Build fisher information matrix evaluator
        hvp_evaluator = gen_hvp_evaluator(
            observations=normalized_obs,
            actions=actions,
            curr_mean=curr_mean,
            curr_log_std=curr_log_std,
            old_mean=old_mean,
            old_log_std=old_log_std,
            policy=self.policy,
            action_scale=self.policy.action_scale,
            action_bias=self.policy.action_bias,
            damping_coeff=self.damping_coeff,
            hvp_subsample=self.hvp_subsample
        )

        # Solve for natural gradient
        npg_grad = conjugate_gradient(
            hvp_evaluator=hvp_evaluator,
            b=policy_grad,
            nsteps=self.FIM_invert_args["iters"],
            residual_tol=self.residual_tol,
        )

        # Compute step size using KL constraint with safety checks
        n_step_size = 2.0 * self.delta
        grad_dot_product = torch.dot(policy_grad, npg_grad).item()

        print(f"grad_dot_product: {grad_dot_product}")

        # If gradient is too small, skip the update
        if abs(grad_dot_product) < 1e-10:
            print("Warning: Gradient too small, skipping update")
            return self._get_stats_dict(
                paths=paths,
                values=values,
                adv=policy_adv,
                dapg_grad=policy_grad,
                npg_grad=npg_grad,
                grad_dot_product=grad_dot_product,
                demo_paths=demo_paths,
                skipped_update=True,
            )

        # Compute step size with clipping
        alpha = np.sqrt(n_step_size / (abs(grad_dot_product) + 1e-8))
        alpha = np.clip(alpha, 0, 1.0)  # Clip step size to prevent too large updates

        print(f"Step size alpha: {alpha}")

        # Update policy
        curr_params = self.get_param_values()
        new_params = curr_params + alpha * npg_grad

        # Set new parameters and update old distribution parameters
        self.update_policy_network(normalized_obs, new_params)

        # update value network through the belmman update for rewards discounted by upcomiing future reweards (next_values)
        returns = rewards + self.gamma * next_values * (1 - dones)
        # Update value network with actual returns
        value_loss = self.update_value_network(normalized_obs, returns)

        # Evaluate update
        _, _, _, new_dist, _ = self.get_current_policy_dist(normalized_obs, actions)
        _, _, _, old_dist, _ = self.get_old_policy_dist(normalized_obs, actions)

        surr_after = compute_surrogate_loss(
            curr_dist=new_dist,
            old_dist=old_dist,
            actions=actions,
            advantages=policy_adv,
            action_scale=self.policy.action_scale,
            action_bias=self.policy.action_bias,
        )
        kl_div = self.compute_kl_divergence(normalized_obs, actions).item()
        # Finally we update the old policy network with the new parmaeters for the next update
        self.set_old_policy()
        # After update, return full stats
        return self._get_stats_dict(
            paths=paths,
            values=values,
            adv=combined_adv,
            policy_grad=npg_grad,
            surr_before=surr_before,
            surr_after=surr_after,
            kl_div=kl_div,
            step_size=alpha,
            grad_dot_product=grad_dot_product,
            value_loss=value_loss,
            demo_paths=demo_paths,
            skipped_update=False,
        )

    def save_model(self, path: str):
        """Save policy and value networks to files"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save policy network
        policy_path = f"{path}_policy.pt"
        torch.save(self.policy.state_dict(), policy_path)

        # Save value network
        value_path = f"{path}_value.pt"
        torch.save(self.value_network.state_dict(), value_path)

        print(f"Models saved to {path}_policy.pt and {path}_value.pt")

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute advantages using GAE"""
        # Use the generalized_advantage_estimate function from functions.py
        advantages = generalized_advantage_estimate(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Normalize advantages (keeping this from original implementation)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _get_stats_dict(
        self,
        paths: List[Dict],
        values: torch.Tensor,
        adv: torch.Tensor,
        policy_grad: Optional[torch.Tensor] = None,
        surr_before: float = 0.0,
        surr_after: float = 0.0,
        kl_div: float = 0.0,
        step_size: float = 0.0,
        grad_dot_product: float = 0.0,
        value_loss: float = 0.0,
        demo_paths: Optional[List[Dict]] = None,
        skipped_update: bool = False,
    ) -> Dict[str, float]:
        """Generate statistics dictionary for logging and debugging.
        Args:
            paths: List of trajectory dictionaries
            values: Value network predictions
            adv: Computed advantages
            policy_grad: Policy gradient (optional)
            surr_before: Surrogate loss before update
            surr_after: Surrogate loss after update
            kl_div: KL divergence after update
            step_size: Step size used for update
            grad_dot_product: Dot product of policy and natural gradients
            value_loss: Value network loss
            demo_paths: Demonstration paths if using DAPG
            skipped_update: Whether update was skipped due to small gradient

        Returns:
            Dictionary containing training statistics
        """
        path_returns = [sum(p["rewards"]) for p in paths]

        return {
            # Training returns
            "mean_return": np.mean(path_returns),
            "std_return": np.std(path_returns),
            "min_return": np.min(path_returns),
            "max_return": np.max(path_returns),
            # Policy update metrics
            "surr_before": surr_before,
            "surr_after": surr_after,
            "surr_improvement": surr_after - surr_before,
            "kl_div": kl_div,
            "step_size": step_size,
            "grad_dot_product": grad_dot_product,
            # Value network metrics
            "value_loss": value_loss,
            "mean_value": values.mean().item(),
            "std_value": values.std().item(),
            # Advantage metrics
            "mean_advantage": adv.mean().item(),
            "std_advantage": adv.std().item(),
            "max_advantage": adv.max().item(),
            "min_advantage": adv.min().item(),
            # Demo related
            "demo_coef": (
                self.lam_0 * (self.lam_1**self.iter_count)
                if demo_paths is not None
                else 0
            ),
            "iter_count": self.iter_count,
            # Gradient metrics
            "policy_grad_norm": torch.norm(policy_grad).item()
            if policy_grad is not None
            else 0.0,
            # Parameters for debugging
            "delta": self.delta,
            "damping_coeff": self.damping_coeff,
            "gamma": self.gamma,
            "skipped_update": skipped_update,
        }

