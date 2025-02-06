import torch
from torch.distributions import Normal
from typing import Optional, Callable
import torch.nn as nn
import numpy as np


def kl_divergence(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between two Gaussian distributions KL(old||new)

    Args:
        old_mean: Mean parameters of old policy distribution
        old_log_std: Log standard deviation of old policy
        new_mean: Mean parameters of new policy distribution
        new_log_std: Log standard deviation of new policy

    Returns:
        Mean KL divergence between the distributions
    """
    old_dist = Normal(old_mean, torch.exp(old_log_std))
    new_dist = Normal(new_mean, torch.exp(new_log_std))

    # Use torch.distributions.kl.kl_divergence
    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)

    return kl.sum(-1).mean()


def compute_surrogate_loss(
    curr_dist: Normal,
    old_dist: Optional[Normal],
    actions: torch.Tensor,
    advantages: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
) -> torch.Tensor:
    """Compute surrogate loss for policy gradient optimization

    Implements importance sampling loss:


    For bounded action spaces, applies tanh transformation and corrects log probs.

    Args:
        curr_dist: Current policy distribution π_new
        old_dist: Previous policy distribution π_old (None on first iteration)
        actions: Actions sampled from old policy
        advantages: Advantage estimates A(s,a)
        action_scale: Scale factor for bounded actions
        action_bias: Bias term for bounded actions

    Returns:
        Tuple of (surrogate_loss, importance_sampling_ratio)
    """
    # Get pre-tanh actions
    normalized_actions = (actions - action_bias) / action_scale
    x_t = torch.atanh(torch.clamp(normalized_actions, -0.999999, 0.999999))

    # Compute current log probabilities
    curr_log_probs = curr_dist.log_prob(x_t)
    log_prob_correction = torch.sum(
        torch.log(torch.clamp(1 - torch.tanh(x_t).pow(2), min=1e-6)), dim=-1
    )
    curr_log_prob = curr_log_probs.sum(-1) - log_prob_correction

    if old_dist is not None:
        # Get old log probabilities
        old_log_probs = old_dist.log_prob(x_t)
        old_log_prob = old_log_probs.sum(-1) - log_prob_correction

        # Compute ratio (relative importance of new policy)
        ratio = torch.exp(curr_log_prob - old_log_prob)
        print("Ratio: ", ratio)
        # Compute surrogate loss
        surrogate_loss = torch.mean(ratio * advantages)
    else:
        ratio = None
        surrogate_loss = torch.mean(curr_log_prob * advantages)

    return surrogate_loss


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Calculate advantages using Generalized Advantage Estimation (GAE)

    From: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING
    GENERALIZED ADVANTAGE ESTIMATION - https://arxiv.org/pdf/1506.02438

    Args:
        rewards (torch.Tensor): shape (T,). rewards[t] is the reward at time t.
        values (torch.Tensor): shape (T+1,). value function predictions for each state
                               (including one extra to bootstrap at time T).
        dones (torch.Tensor): shape (T,). dones[t] = 1 if the episode terminates at step t,
                             otherwise 0. (Set dones[t]=1 for each boundary between episodes)
        gamma (float): discount factor
        gae_lambda (float): GAE parameter (often denoted by λ in literature)

    Returns:
        advantages (torch.Tensor): shape (T,). advantage estimates for each time step.

    Notes:
        - If you have multiple episodes in a single array, ensure that `dones[t] = 1` exactly
          at the end of each episode. This resets the advantage calculation so it does not
          bootstrap from the next episode.
        - `values.shape` = (T+1,) so that `values[t+1]` is valid up to t = T-1.
    """
    advantages = torch.zeros_like(rewards)
    not_done = 1.0 - dones
    gae = 0.0

    for t in reversed(range(len(rewards))):
        # values[t+1] will work now because we included the next value
        delta = rewards[t] + gamma * values[t + 1] * not_done[t] - values[t]
        gae = delta + gamma * gae_lambda * not_done[t] * gae
        advantages[t] = gae

    return advantages


def conjugate_gradient(
    hvp_evaluator: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Solve Ax = b using conjugate gradient algorithm

    Used to compute natural gradient direction by solving:
    F(θ)x = g
    where F is the Fisher Information Matrix and g is the policy gradient.

    Avoids explicitly forming F by using Hessian-vector products.

    Args:
        hvp_evaluator: Function computing product of FIM with vector
        b: Right hand side vector (typically policy gradient)
        nsteps: Maximum conjugate gradient iterations
        residual_tol: Tolerance for early stopping

    Returns:
        Solution x to Ax = b
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
    """Compute Fisher Information Matrix vector product

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
    """Generate HVP evaluator function"""
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
    """Compute policy gradient using automatic differentiation

    Computes ∇_θ L(θ) where L is the policy objective.
    Flattens gradients for use with natural gradient computation.

    Args:
        loss: Policy objective loss
        policy: Policy network

    Returns:
        Flattened policy gradient vector
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


def mean_log_likelihood(model, log_std, action_dim, observations, actions):
    """Compute mean and log likelihood of actions under policy

    Args:
        actions: Actions in post-tanh space [-1,1]
        mean: Mean in pre-tanh space
        log_std: Log standard deviation in pre-tanh space

    Returns:
        mean: Policy mean
        log_likelihood: Log likelihood of the actions
    """
    mean = model(observations)
    zs = (actions - mean) / torch.exp(log_std)
    log_likelihood = (
        -0.5 * torch.sum(zs**2, dim=1)
        - torch.sum(log_std)
        - 0.5 * action_dim * np.log(2 * np.pi)
    )
    return mean, log_likelihood


def vanilla_advantage(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return rewards - values.detach()


def normalize_tensors(tesnors: torch.Tensor) -> torch.Tensor:
    return (tesnors - tesnors.mean()) / (tesnors.std() + 1e-8)
