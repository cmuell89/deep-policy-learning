import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict, List, Optional
import numpy as np
import copy
import os


class DAPG:
    def __init__(
        self,
        policy_network: nn.Module,
        value_network, 
        normalized_step_size=0.01,  # This is delta in our implementation
        FIM_invert_args={'iters': 10, 'damping': 1e-4},
        hvp_sample_frac=1.0,
        kl_dist=None,
        lam_0=1.0,  # demo coef
        lam_1=0.95,  # decay coef
        **kwargs,
    ):
        """
        Demonstration Assisted Natural Policy Gradient implementation
        
        Args:
            policy_network: Neural network for the policy
            baseline: Neural network for value function
            demo_paths: List of demonstration paths
            normalized_step_size: Normalized step size for KL divergence constraint
            FIM_invert_args: Arguments for Fisher Information Matrix inversion
            hvp_sample_frac: Subsampling fraction for Hessian-vector product calculation
            seed: Random seed for reproducibility
            save_logs: Boolean to save logs
            kl_dist: KL divergence constraint
            lam_0: Damping coefficient for demonstration paths
            lam_1: Decay factor for demonstration paths
        """
        self.policy = policy_network
        self.value_network = value_network  # Using baseline as value network
        self.value_learning_rate = 1e-3  # Default value learning rate
        self.gamma = 0.99  # Default gamma
        self.delta = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.damping_coeff = FIM_invert_args.get('damping', 1e-4)
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0
        self.hvp_subsample = hvp_sample_frac
        self.input_normalization = 0.9  # Default value
        self.FIM_invert_args = FIM_invert_args
        
        # Only use optimizer for value network since policy uses natural gradient
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_learning_rate)
        self.old_means = None
        self.old_log_stds = None
        
        # For observation normalization
        self.obs_mean = None
        self.obs_std = None

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # temporal difference at time t
            # rewards[t] is the immediate reward
            # self.gamma is the discount factor (typically 0.99)
            #next_value is V(s_{t+1}) (0 if terminal state)
            #(1 - dones[t]) zeros out future value if episode is done
            #values[t] is V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            # delta is the current TD error
            # self.gamma * (1 - dones[t]) is the discount factor (zeroed if done)
            # last_advantage accumulates future advantages
            advantages[t] = delta + self.gamma * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
        return advantages


    def compute_kl_divergence(self, states: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between old and current policy distributions"""
        with torch.no_grad():
            _, info = self.policy(states)
            new_means = info['mean']
            new_log_stds = info['log_std']
            
            if self.old_log_stds is None:
                self.old_log_stds = new_log_stds
                self.old_means = new_means
                return torch.tensor(0.0, device=states.device)
            
            # Convert log_stds to stds
            old_std = torch.exp(self.old_log_stds)
            new_std = torch.exp(new_log_stds)
            
            # Compute KL divergence
            Nr = (self.old_means - new_means) ** 2 + old_std ** 2 - new_std ** 2
            Dr = 2 * new_std ** 2 + 1e-8
            
            kl = torch.sum(Nr / Dr + new_log_stds - self.old_log_stds, dim=-1)
            kl = kl.mean()
            
            return kl.clamp(0.0, 0.1)

    def conjugate_gradient(self, hvp_evaluator, b, nsteps: int = 10) -> torch.Tensor:
        """Conjugate gradient algorithm with improved numerical stability"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        
        # Early exit if b is zero
        if torch.allclose(b, torch.zeros_like(b)):
            return x
        
        r_norm_sq = torch.dot(r, r)
        
        for i in range(nsteps):
            # Compute Hessian-vector product
            Ap = hvp_evaluator(p)
            
            # Check for numerical issues
            if not torch.isfinite(Ap).all():
                print(f"Warning: Non-finite values in CG iteration {i}")
                break
            
            pAp = torch.dot(p, Ap)
            
            # Check for numerical issues in denominator
            if abs(pAp) < 1e-8:
                print(f"Warning: Small pAp value in CG iteration {i}")
                break
            
            alpha = r_norm_sq / pAp
            
            # Clip alpha to prevent extreme steps
            alpha = torch.clamp(alpha, -100, 100)
            
            x_new = x + alpha * p
            
            # Check if update is too large
            if torch.norm(x_new - x) > 100:
                print(f"Warning: Large update in CG iteration {i}")
                break
            
            x = x_new
            r_new = r - alpha * Ap
            r_norm_sq_new = torch.dot(r_new, r_new)
            
            # Check for convergence
            if r_norm_sq_new < 1e-10:
                break
            
            beta = r_norm_sq_new / r_norm_sq
            r_norm_sq = r_norm_sq_new
            r = r_new
            p = r + beta * p
        
        return x

    def compute_surrogate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_dist: Normal
    ) -> torch.Tensor:
        """Compute surrogate loss for policy optimization"""
        # Get new distribution parameters from current policy
        _, info = self.policy(states)
        mean = info['mean']
        log_std = info.get('log_std', torch.zeros_like(mean))  # Default if not provided
        std = torch.exp(log_std)
        
        # Create new distribution
        new_dist = Normal(mean, std)
        
        # Sum log probs across action dimensions
        log_ratio = new_dist.log_prob(actions).sum(-1) - old_dist.log_prob(actions).sum(-1)
        ratio = torch.exp(log_ratio)
        
        # Compute surrogate loss
        surr_loss = -(ratio * advantages).mean()
        return surr_loss

    def compute_policy_gradient(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient using the advantages"""
        # Get distribution parameters from current policy
        _, info = self.policy(states)
        mean = info['mean']
        log_std = info.get('log_std', torch.zeros_like(mean))
        std = torch.exp(log_std)
        
        # Create distribution
        dist = Normal(mean, std)
        
        # Sum log probs across action dimensions
        log_probs = dist.log_prob(actions).sum(-1)  # Sum across action dimensions
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute gradients with allow_unused=True
        policy_grad = torch.autograd.grad(
            policy_loss, 
            self.policy.parameters(),
            allow_unused=True,
            create_graph=False,
            retain_graph=False
        )
        
        # Handle unused parameters by replacing None gradients with zeros
        policy_grad = [torch.zeros_like(param) if grad is None else grad 
                      for grad, param in zip(policy_grad, self.policy.parameters())]
        
        policy_grad = torch.cat([grad.view(-1) for grad in policy_grad])
        
        return policy_grad

    def fisher_information(self, observations, actions, vector, regu_coef=1e-3):
        """Compute Hessian-vector product with improved numerical stability"""
        device = observations.device
        vec = vector.to(device)
        
        # Optional subsampling with stability check
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            sample_size = max(int(self.hvp_subsample * num_samples), 1)
            rand_idx = np.random.choice(num_samples, size=sample_size)
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions

        # Ensure requires_grad is True for policy parameters
        for param in self.policy.parameters():
            param.requires_grad_(True)
            
        try:
            # Store current requires_grad state
            prev_grad_state = {}
            for name, param in self.policy.named_parameters():
                prev_grad_state[name] = param.requires_grad
                param.requires_grad_(True)
            
            # Debug prints
            print("\nGradient tracking status:")
            for name, param in self.policy.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")
            
            # Compute KL divergence using existing method but without clamping
            with torch.set_grad_enabled(True):
                _, info = self.policy(obs)
                new_means = info['mean']
                new_log_stds = info['log_std']
                
                # Convert log_stds to stds
                old_std = torch.exp(self.old_log_stds)
                new_std = torch.exp(new_log_stds)
                
                # Compute KL divergence without clamping
                Nr = (self.old_means - new_means) ** 2 + old_std ** 2 - new_std ** 2
                Dr = 2 * new_std ** 2 + 1e-8
                kl = torch.sum(Nr / Dr + new_log_stds - self.old_log_stds, dim=-1).mean()
                
                print(f"KL value: {kl.item()}")
                print(f"KL requires grad: {kl.requires_grad}")
            
            # Compute gradient of KL
            grad = torch.autograd.grad(
                kl,
                self.policy.parameters(),
                create_graph=True,
                allow_unused=True,
                retain_graph=True
            )
            
            # Debug print gradient values
            print("\nGradient values:")
            for g, (name, param) in zip(grad, self.policy.named_parameters()):
                if g is not None:
                    print(f"{name}: {g.norm().item()}")
                else:
                    print(f"{name}: None")
            
            # Filter out None gradients and flatten
            grad = [g if g is not None else torch.zeros_like(p) 
                   for g, p in zip(grad, self.policy.parameters())]
            flat_grad = torch.cat([g.view(-1) for g in grad])
            
            # Compute gradient-vector product
            grad_vec_prod = torch.sum(flat_grad * vec)
            
            # Compute Hessian-vector product
            hvp = torch.autograd.grad(
                grad_vec_prod,
                self.policy.parameters(),
                retain_graph=False
            )
            
            # Filter out None gradients and flatten
            hvp = [g if g is not None else torch.zeros_like(p) 
                  for g, p in zip(hvp, self.policy.parameters())]
            hvp_flat = torch.cat([g.view(-1) for g in hvp])
            
            # Restore previous requires_grad state
            for name, param in self.policy.named_parameters():
                param.requires_grad_(prev_grad_state[name])
            
            return hvp_flat + regu_coef * vec
            
        except RuntimeError as e:
            print(f"Warning: Runtime error in HVP computation: {e}")
            # Restore previous requires_grad state
            for name, param in self.policy.named_parameters():
                param.requires_grad_(prev_grad_state[name])
            return regu_coef * vec

    def gen_fim_evaluator(self, observations, actions, hvp_subsample=None):
        """Build a function that evaluates HVP for given inputs"""
        def eval(vector):
            return self.fisher_information(observations, actions, vector, self.damping_coeff)
        return eval

    def update_value_network(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """Update value network to better predict returns"""
        for _ in range(5):  # Multiple epochs of value fitting
            value_pred = self.value_network(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
        return value_loss.item()

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation with improved stability"""
        # Add GAE lambda parameter as class attribute if not present
        if not hasattr(self, 'gae_lambda'):
            self.gae_lambda = 0.95  # Standard GAE lambda value
        
        # Compute deltas with value clipping for stability
        deltas = (
            rewards + 
            self.gamma * next_values * (1 - dones) - 
            values
        ).clamp(-10, 10)  # Clip TD errors
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            # Compute advantage with lambda and proper discounting
            advantages[t] = deltas[t] + \
                           self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages to prevent extreme values
        advantages = advantages.clamp(-10, 10)
        
        return advantages

    def update_normalization_stats(self, observations: torch.Tensor):
        """Update running statistics for input normalization with improved numerical stability"""
        with torch.no_grad():
            # Compute batch statistics with epsilon to prevent division by zero
            batch_mean = observations.mean(dim=0)
            batch_std = observations.std(dim=0, unbiased=False) + 1e-8
            
            # Initialize stats if first time
            if self.obs_mean is None:
                self.obs_mean = batch_mean
                self.obs_std = batch_std
            else:
                # Update running statistics using exponential moving average
                self.obs_mean = (self.input_normalization * self.obs_mean + 
                               (1 - self.input_normalization) * batch_mean)
                self.obs_std = (self.input_normalization * self.obs_std + 
                              (1 - self.input_normalization) * batch_std)
            
            # Ensure standard deviation is never too close to zero
            self.obs_std = torch.clamp(self.obs_std, min=1e-8)

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics"""
        if self.obs_mean is None or self.input_normalization == 0:
            return observations
        
        try:
            # Normalize with broadcasting
            normalized = (observations - self.obs_mean) / self.obs_std
            normalized = torch.clamp(normalized, -10.0, 10.0)
            
            if torch.isnan(normalized).any():
                print("Warning: NaN values after normalization")
                return observations
            
            return normalized
            
        except Exception as e:
            print(f"Warning: Error in normalization: {e}")
            return observations
        
    def get_param_values(self):
        return torch.cat([param.view(-1) for param in self.policy.parameters()])

    def set_param_values(self, normalized_obs, new_params, set_old=True):
        """Add debug prints to track parameter updates"""
        if set_old:
            # Store old distribution parameters
            with torch.no_grad():
                _, info = self.policy(normalized_obs)
                print("Old means before update:", self.old_means.mean().item())
                print("Old log_stds before update:", self.old_log_stds.mean().item())
                self.old_means = info['mean']
                self.old_log_stds = info['log_std']
                print("Old means after update:", self.old_means.mean().item())
                print("Old log_stds after update:", self.old_log_stds.mean().item())
        
        # Update parameters
        idx = 0
        for param in self.policy.parameters():
            param_size = param.numel()
            new_param_slice = new_params[idx:idx + param_size].view(param.size())
            print(f"Parameter update diff: {(param.data - new_param_slice).abs().mean().item()}")
            param.data = new_param_slice
            idx += param_size


    def update(self, paths: List[Dict], demo_paths: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Update policy following the DAPG algorithm structure"""
        device = next(self.policy.parameters()).device
        
        # Concatenate trajectory data and move to correct device
        observations = torch.FloatTensor(np.concatenate([path["observations"] for path in paths])).to(device)
        actions = torch.FloatTensor(np.concatenate([path["actions"] for path in paths])).to(device)
        rewards = torch.FloatTensor(np.concatenate([path["rewards"] for path in paths])).to(device)
        dones = torch.FloatTensor(np.concatenate([path["dones"] for path in paths])).to(device)
        
        # Use update_normalization_stats and normalize_observations methods
        self.update_normalization_stats(observations)
        normalized_obs = self.normalize_observations(observations)
        
        # Get policy distribution parameters from normalized observations
        with torch.no_grad():
            _, info = self.policy(normalized_obs)
            self.old_means = info['mean']
            self.old_log_stds = info['log_std']
        
        # Compute values and returns for value network update
        with torch.no_grad():
            values = self.value_network(normalized_obs).squeeze()
            next_values = torch.zeros_like(values)
            next_values[:-1] = values[1:]
        
        # Use compute_advantages instead of direct GAE computation
        advantages = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Handle demonstration paths with proper scaling
        if demo_paths is not None and self.lam_0 > 0.0:
            demo_obs = np.concatenate([path["observations"] for path in demo_paths])
            demo_act = np.concatenate([path["actions"] for path in demo_paths])
            demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            self.iter_count += 1
            
            # Normalize demo observations too
            demo_obs_tensor = torch.FloatTensor(demo_obs).to(device)
            normalized_demo_obs = self.normalize_observations(demo_obs_tensor)
            
            # Concatenate all data with proper scaling
            all_obs = torch.cat([normalized_obs, normalized_demo_obs])
            all_act = torch.FloatTensor(np.concatenate([actions.cpu().numpy(), demo_act])).to(device)
            all_adv = torch.FloatTensor(np.concatenate([
                advantages.cpu().numpy()/(advantages.std().cpu().numpy() + 1e-8),
                demo_adv
            ])).to(device) * 1e-2
        else:
            all_obs = normalized_obs
            all_act = actions
            all_adv = advantages

        # Compute surrogate loss before update
        surr_before = self.compute_surrogate_loss(
            normalized_obs, actions, advantages, 
            Normal(self.old_means, self.old_log_stds.exp())  # Convert log_std to std
        ).item()

        # Compute DAPG gradient with proper scaling
        sample_coef = all_adv.shape[0]/advantages.shape[0]
        policy_grad = sample_coef * self.compute_policy_gradient(all_obs, all_act, all_adv)
        
        # Build fisher information matrix evaluator
        fisher_info_eval = self.gen_fim_evaluator(
            observations=normalized_obs,
            actions=actions,
            hvp_subsample=self.hvp_subsample
        )
        
        # Solve for natural gradient
        npg_grad = self.conjugate_gradient(
            fisher_info_eval,
            policy_grad,
            nsteps=self.FIM_invert_args['iters']
        )
        
        # Compute step size using KL constraint with safety checks
        n_step_size = 2.0 * self.delta
        grad_dot_product = torch.dot(policy_grad, npg_grad).item()
        
        print(f"grad_dot_product: {grad_dot_product}")
        
        # If gradient is too small, skip the update
        if abs(grad_dot_product) < 1e-8:
            print("Warning: Gradient too small, skipping update")
            # Return full stats dictionary with default values
            path_returns = [sum(p["rewards"]) for p in paths]
            return {
                # Training returns
                'mean_return': np.mean(path_returns),
                'std_return': np.std(path_returns),
                'min_return': np.min(path_returns),
                'max_return': np.max(path_returns),
                
                # Policy update metrics
                'surr_before': 0.0,
                'surr_after': 0.0,
                'surr_improvement': 0.0,
                'kl_div': 0.0,
                'step_size': 0.0,
                'grad_dot_product': grad_dot_product,
                
                # Value network metrics
                'value_loss': 0.0,
                'mean_value': values.mean().item(),
                'std_value': values.std().item(),
                
                # Advantage metrics
                'mean_advantage': advantages.mean().item(),
                'std_advantage': advantages.std().item(),
                'max_advantage': advantages.max().item(),
                'min_advantage': advantages.min().item(),
                
                # Demo related
                'demo_coef': self.lam_0 * (self.lam_1 ** self.iter_count) if demo_paths is not None else 0,
                'iter_count': self.iter_count,
                
                # Gradient metrics
                'policy_grad_norm': torch.norm(policy_grad).item(),
                'npg_grad_norm': torch.norm(npg_grad).item(),
                
                # Parameters for debugging
                'delta': self.delta,
                'damping_coeff': self.damping_coeff,
                'gamma': self.gamma,
                'skipped_update': True
            }
            
        # Compute step size with clipping
        alpha = np.sqrt(n_step_size / (abs(grad_dot_product) + 1e-8))
        alpha = np.clip(alpha, 0, 1.0)  # Clip step size to prevent too large updates
        
        print(f"Step size alpha: {alpha}")
        
        # Update policy
        curr_params = self.get_param_values()
        new_params = curr_params + alpha * npg_grad
        
        # Set new parameters and update old distribution parameters
        self.set_param_values(normalized_obs, new_params, set_old=True)

        #update value network through the belmman update for rewards discounted by upcomiing future reweards (next_values)
        returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Update value network with actual returns
        value_loss = self.update_value_network(normalized_obs, returns)

        # Evaluate update
        surr_after = self.compute_surrogate_loss(
            normalized_obs, actions, advantages,
            Normal(self.old_means, self.old_log_stds.exp())
        ).item()
        kl_div = self.compute_kl_divergence(normalized_obs).item()

        # Update running statistics
        path_returns = [sum(p["rewards"]) for p in paths]
        stats = {
            # Training returns
            'mean_return': np.mean(path_returns),
            'std_return': np.std(path_returns),
            'min_return': np.min(path_returns),
            'max_return': np.max(path_returns),
            
            # Policy update metrics
            'surr_before': surr_before,
            'surr_after': surr_after,
            'surr_improvement': surr_after - surr_before,
            'kl_div': kl_div,
            'step_size': alpha,
            'grad_dot_product': grad_dot_product,
            
            # Value network metrics
            'value_loss': value_loss,
            'mean_value': values.mean().item(),
            'std_value': values.std().item(),
            
            # Advantage metrics
            'mean_advantage': advantages.mean().item(),
            'std_advantage': advantages.std().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
            
            # Demo related (if using demonstrations)
            'demo_coef': self.lam_0 * (self.lam_1 ** self.iter_count) if demo_paths is not None else 0,
            'iter_count': self.iter_count,
            
            # Gradient metrics
            'policy_grad_norm': torch.norm(policy_grad).item(),
            'npg_grad_norm': torch.norm(npg_grad).item(),
            
            # Parameters for debugging
            'delta': self.delta,
            'damping_coeff': self.damping_coeff,
            'gamma': self.gamma,
        }
        
        # Print key metrics
        print(f"\nUpdate Stats:")
        print(f"KL Divergence: {stats['kl_div']:.6f}")
        print(f"Step Size: {stats['step_size']:.6f}")
        print(f"Surrogate Improvement: {stats['surr_improvement']:.6f}")
        print(f"Value Loss: {stats['value_loss']:.6f}")
        print(f"Mean Advantage: {stats['mean_advantage']:.6f}")
        if demo_paths is not None:
            print(f"Demo Coefficient: {stats['demo_coef']:.6f}")
        
        return stats

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

    
