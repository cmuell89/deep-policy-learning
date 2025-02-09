import torch
from torch.distributions import Normal
from typing import Optional, Callable, Tuple
import torch.nn as nn
import numpy as np


def kl_divergence(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussian distributions KL(old||new).

    Args:
        old_mean: Mean parameters of old policy distribution
        old_log_std: Log standard deviation of old policy
        new_mean: Mean parameters of new policy distribution
        new_log_std: Log standard deviation of new policy

    Returns:
        torch.Tensor: Mean KL divergence between the distributions
    """
    old_dist = Normal(old_mean, torch.exp(old_log_std))
    new_dist = Normal(new_mean, torch.exp(new_log_std))

    # Use torch.distributions.kl.kl_divergence
    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)

    return kl.sum(-1).mean()


def compute_surrogate_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """
    Compute PPO surrogate loss from log probabilities and advantages.

    Args:
        advantages: Advantage estimates A(s,a)
        log_probs: Current policy log probabilities
        old_log_probs: Previous policy log probabilities (None on first iteration)
        clip_epsilon: PPO clipping parameter

    Returns:
        torch.Tensor: PPO surrogate loss value
    """
    if old_log_probs is not None:
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        # Take minimum of surrogate terms (pessimistic bound)
        surrogate_loss = -torch.min(surr1, surr2).mean()
    else:
        # First iteration - no clipping
        surrogate_loss = -(log_probs * advantages).mean()

    return surrogate_loss  # Only return the loss


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages using Generalized Advantage Estimation (GAE).

    From: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING
    GENERALIZED ADVANTAGE ESTIMATION - https://arxiv.org/pdf/1506.02438

    Args:
        rewards: Shape (T,). rewards[t] is the reward at time t
        values: Shape (T+1,). Value function predictions for each state
               (including one extra to bootstrap at time T)
        dones: Shape (T,). dones[t] = 1 if episode terminates at step t,
              otherwise 0. (Set dones[t]=1 for each boundary between episodes)
        gamma: Discount factor
        gae_lambda: GAE parameter (often denoted by λ in literature)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Returns and advantages for each time step
            - returns: Shape (T,). Returns for each time step
            - advantages: Shape (T,). Advantage estimates for each time step

    Notes:
        - If you have multiple episodes in a single array, ensure that `dones[t] = 1` exactly
          at the end of each episode. This resets the advantage calculation so it does not
          bootstrap from the next episode.
        - `values.shape` = (T+1,) so that `values[t+1]` is valid up to t = T-1.
    """
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(len(rewards))):
        # If done[t]==1, mask = 0; if done[t]==0, mask=1
        mask = 1.0 - dones[t]

        # Same GAE logic, just use `mask` here:
        running_returns = rewards[t] + gamma * running_returns * mask
        td_error = rewards[t] + gamma * previous_value * mask - values.data[t]
        running_advants = td_error + gamma * gae_lambda * running_advants * mask

        returns[t] = running_returns
        advantages[t] = running_advants
        previous_value = values.data[t]

    return returns, advantages


def conjugate_gradient(
    hvp_evaluator: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """
    Solve Ax = b using conjugate gradient algorithm.

    Used to compute natural gradient direction by solving:
    F(θ)x = g
    where F is the Fisher Information Matrix and g is the policy gradient.

    Args:
        hvp_evaluator: Function computing product of FIM with vector
        b: Right hand side vector (typically policy gradient)
        nsteps: Maximum conjugate gradient iterations
        residual_tol: Tolerance for early stopping

    Returns:
        torch.Tensor: Solution x to Ax = b
    """
    device = b.device
    x = torch.zeros_like(b, device=device)
    r = b.clone()
    p = r.clone()

    rdotr = torch.dot(r, r)
    print("\nCG Initial state:")
    print(f"Initial residual norm: {torch.sqrt(rdotr)}")
    print(f"RHS vector (b) norm: {torch.norm(b)}")

    for i in range(nsteps):
        print(f"\nCG Iteration {i + 1}/{nsteps}")

        # Compute Hessian-vector product
        Hp = hvp_evaluator(p)
        print(f"HVP norm: {torch.norm(Hp)}")
        print(f"Direction (p) norm: {torch.norm(p)}")

        # Compute step size
        pHp = torch.dot(p, Hp)
        print(f"pHp value: {pHp}")

        alpha = rdotr / (pHp + 1e-8)
        print(f"Step size (alpha): {alpha}")

        # Update solution and residual
        x_old = x.clone()
        x = x + alpha * p
        print(f"Solution change: {torch.norm(x - x_old)}")
        print(f"Current solution norm: {torch.norm(x)}")

        r = r - alpha * Hp
        newrdotr = torch.dot(r, r)
        print(f"New residual norm: {torch.sqrt(newrdotr)}")

        # Early stopping check
        if newrdotr < residual_tol:
            print(f"Converged at iteration {i + 1}, residual: {torch.sqrt(newrdotr)}")
            break

        beta = newrdotr / (rdotr + 1e-8)
        print(f"Beta: {beta}")

        # Update conjugate direction
        p = r + beta * p
        rdotr = newrdotr

    return x


def compute_fisher_information(
    observations: torch.Tensor,
    actions: torch.Tensor,
    curr_mean: torch.Tensor,
    curr_log_std: torch.Tensor,
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    vector: torch.Tensor,
    policy: nn.Module,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    damping_coeff: float,
    hvp_subsample: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute Fisher Information Matrix vector product.

    Args:
        observations: States/observations tensor
        actions: Actions tensor
        curr_mean: Current policy mean
        curr_log_std: Current policy log std
        old_mean: Old policy mean
        old_log_std: Old policy log std
        vector: Vector to compute product with
        policy: Policy network
        action_scale: Action scaling factor
        action_bias: Action bias
        damping_coeff: Regularization coefficient
        hvp_subsample: Optional subsampling ratio for HVP computation

    Returns:
        torch.Tensor: Fisher Information Matrix vector product
    """
    device = observations.device
    try:
        trainable_params = list(policy.parameters()) + [curr_log_std]
        # Compute KL divergence
        mean_kl = kl_divergence(old_mean, old_log_std, curr_mean, curr_log_std)

        # Get gradient of KL
        grad_kl = torch.autograd.grad(mean_kl, trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_kl])

        # Compute Hessian-vector product
        grad_vector_prod = torch.sum(flat_grad * vector)
        hvp = torch.autograd.grad(grad_vector_prod, trainable_params, create_graph=True)
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat + damping_coeff * vector

    except Exception as e:
        print(f"Device: {device}")
        print(f"observations device: {observations.device}")
        print(f"actions device: {actions.device}")
        print(f"vector device: {vector.device}")
        print(f"policy device: {next(policy.parameters()).device}")
        raise e


def gen_hvp_evaluator(
    observations: torch.Tensor,
    actions: torch.Tensor,
    curr_mean: torch.Tensor,
    curr_log_std: torch.Tensor,
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    policy: nn.Module,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    damping_coeff: float,
    hvp_subsample: Optional[float] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Generate Hessian-vector product evaluator function.

    Args:
        observations: States/observations tensor
        actions: Actions tensor
        curr_mean: Current policy mean
        curr_log_std: Current policy log std
        old_mean: Old policy mean
        old_log_std: Old policy log std
        policy: Policy network
        action_scale: Action scaling factor
        action_bias: Action bias
        damping_coeff: Regularization coefficient
        hvp_subsample: Optional subsampling ratio for HVP computation

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Function that computes HVP with given vector
    """
    device = observations.device

    def evaluator(vector: torch.Tensor) -> torch.Tensor:
        return compute_fisher_information(
            observations=observations,
            actions=actions,
            curr_mean=curr_mean,
            curr_log_std=curr_log_std,
            old_mean=old_mean,
            old_log_std=old_log_std,
            vector=vector.to(device),
            policy=policy,
            action_scale=action_scale,
            action_bias=action_bias,
            damping_coeff=damping_coeff,
            hvp_subsample=hvp_subsample,
        )

    return evaluator


def compute_policy_gradient(
    loss: torch.Tensor,
    policy: nn.Module,
) -> torch.Tensor:
    """
    Compute policy gradient using automatic differentiation.

    Computes ∇_θ L(θ) where L is the policy objective.
    Flattens gradients for use with natural gradient computation.

    Args:
        loss: Policy objective loss
        policy: Policy network

    Returns:
        torch.Tensor: Flattened policy gradient vector
    """
    device = loss.device

    # Zero gradients
    for param in policy.parameters():
        if param.grad is not None:
            param.grad.zero_()

    # Compute gradients
    grads = torch.autograd.grad(
        loss, policy.parameters(), create_graph=True, retain_graph=True
    )

    # Flatten gradients with device consistency
    flat_grad = torch.cat([grad.view(-1) for grad in grads]).to(device)

    return flat_grad


def mean_log_likelihood(
    model: nn.Module,
    log_std: torch.Tensor,
    action_dim: int,
    observations: torch.Tensor,
    actions: torch.Tensor,
    bounded: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and log likelihood of actions under policy.

    Args:
        model: Policy network model
        log_std: Log standard deviations of policy
        action_dim: Dimension of action space
        observations: Batch of observations
        actions: Batch of actions
        bounded: Whether actions are bounded (True) or unbounded (False)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - mean: Policy mean for given observations
            - log_likelihood: Log likelihood of actions under policy
    """
    # Get mean from policy network
    mean = model(observations)

    # Handle shapes as before...
    if action_dim == 1:
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if log_std.dim() == 0:
            log_std = log_std.unsqueeze(0)

    if log_std.shape != mean.shape:
        log_std = log_std.expand(mean.shape[0], -1)

    if bounded:
        # For bounded actions, we need to:
        # 1. Transform actions back to pre-tanh space
        normalized_actions = (actions - model.out_bias) / model.out_scale
        x_t = torch.atanh(torch.clamp(normalized_actions, -0.999999, 0.999999))

        # 2. Compute Gaussian log likelihood in pre-tanh space
        std = torch.exp(log_std)
        zs = (x_t - mean) / (std + 1e-8)
        log_likelihood = (
            -0.5 * zs.pow(2).sum(-1)
            - log_std.sum(-1)
            - 0.5 * action_dim * np.log(2 * np.pi)
        )

        # 3. Add tanh Jacobian correction
        log_likelihood = log_likelihood - torch.sum(
            torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6), dim=-1
        )
    else:
        # For unbounded actions (like CartPole), just use raw Gaussian
        std = torch.exp(log_std)
        zs = (actions - mean) / (std + 1e-8)
        log_likelihood = (
            -0.5 * zs.pow(2).sum(-1)
            - log_std.sum(-1)
            - 0.5 * action_dim * np.log(2 * np.pi)
        )

    return mean, log_likelihood


def vanilla_advantage(
    rewards: torch.Tensor, 
    values: torch.Tensor
) -> torch.Tensor:
    """
    Compute vanilla advantage estimates as returns minus values.

    Args:
        rewards: Tensor of rewards
        values: Tensor of value estimates

    Returns:
        torch.Tensor: Advantage estimates
    """
    return rewards - values
