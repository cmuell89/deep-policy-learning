import unittest
import torch
import torch.nn as nn
from torch.distributions import Normal
from lr_challenge.learning.functions import (
    kl_divergence,
    compute_surrogate_loss,
    generalized_advantage_estimate,
    conjugate_gradient,
    compute_fisher_information,
    gen_hvp_evaluator,
    compute_policy_gradient,
    mean_log_likelihood,
)


class SimplePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)

        # Common dimensions
        self.batch_size = 10
        self.obs_dim = 4
        self.action_dim = 2

        # Create sample data
        self.observations = torch.randn(self.batch_size, self.obs_dim)
        self.actions = torch.randn(self.batch_size, self.action_dim)
        self.advantages = torch.randn(self.batch_size)

        # Create policy
        self.policy = SimplePolicy(self.obs_dim, self.action_dim).to(self.device)

        # Distribution parameters
        self.old_mean = torch.zeros(self.batch_size, self.action_dim)
        self.old_log_std = torch.zeros(self.action_dim)
        self.new_mean = torch.ones(self.batch_size, self.action_dim)
        self.new_log_std = torch.zeros(self.action_dim)

    def test_kl_divergence(self):
        """Test KL divergence computation between two Gaussian distributions"""
        kl = kl_divergence(
            self.old_mean, self.old_log_std, self.new_mean, self.new_log_std
        )

        # KL should be positive
        self.assertGreater(kl.item(), 0)

        # KL should be zero for identical distributions
        kl_same = kl_divergence(
            self.old_mean, self.old_log_std, self.old_mean, self.old_log_std
        )
        self.assertAlmostEqual(kl_same.item(), 0, places=5)

    def test_compute_surrogate_loss(self):
        """Test surrogate loss computation"""
        # Create distributions
        curr_dist = Normal(self.new_mean, torch.exp(self.new_log_std))
        old_dist = Normal(self.old_mean, torch.exp(self.old_log_std))

        # Test with action scaling
        action_scale = torch.ones(self.action_dim)
        action_bias = torch.zeros(self.action_dim)

        loss = compute_surrogate_loss(
            curr_dist,
            old_dist,
            self.actions,
            self.advantages,
            action_scale,
            action_bias,
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # Should be scalar

    def test_generalized_advantage_estimate(self):
        """Test GAE computation"""
        T = 5  # Timeline length
        rewards = torch.randn(T)
        values = torch.randn(T + 1)  # Include bootstrap value
        dones = torch.zeros(T)
        gamma = 0.99
        gae_lambda = 0.95

        returns, advantages = generalized_advantage_estimate(
            rewards, values, dones, gamma, gae_lambda
        )

        self.assertEqual(advantages.shape, rewards.shape)
        self.assertTrue(torch.all(torch.isfinite(advantages)))

    def test_conjugate_gradient(self):
        """Test conjugate gradient solver"""
        n = 10
        A = torch.randn(n, n)
        A = A.T @ A  # Make positive definite
        b = torch.randn(n)

        def hvp_evaluator(v):
            return A @ v

        x = conjugate_gradient(hvp_evaluator, b, nsteps=10)

        # Check solution quality
        residual = torch.norm(A @ x - b)
        self.assertLess(residual, 1e-4)

    def test_compute_fisher_information(self):
        """Test Fisher Information Matrix vector product computation"""
        vector = torch.ones(sum(p.numel() for p in self.policy.parameters()))

        fim_product = compute_fisher_information(
            self.observations,
            self.actions,
            self.new_mean,
            self.new_log_std,
            self.old_mean,
            self.old_log_std,
            vector,
            self.policy,
            torch.ones(self.action_dim),
            torch.zeros(self.action_dim),
            damping_coeff=0.1,
        )

        self.assertEqual(fim_product.shape, vector.shape)

    def test_gen_hvp_evaluator(self):
        """Test HVP evaluator generation"""
        evaluator = gen_hvp_evaluator(
            self.observations,
            self.actions,
            self.new_mean,
            self.new_log_std,
            self.old_mean,
            self.old_log_std,
            self.policy,
            torch.ones(self.action_dim),
            torch.zeros(self.action_dim),
            damping_coeff=0.1,
        )

        vector = torch.ones(sum(p.numel() for p in self.policy.parameters()))
        result = evaluator(vector)

        self.assertEqual(result.shape, vector.shape)

    def test_compute_policy_gradient(self):
        """Test policy gradient computation"""
        # Create dummy loss
        output = self.policy(self.observations)
        loss = output.mean()

        grad = compute_policy_gradient(loss, self.policy)

        self.assertEqual(
            grad.shape, torch.Size([sum(p.numel() for p in self.policy.parameters())])
        )

    def test_mean_log_likelihood(self):
        """Test mean and log likelihood computation"""
        mean, log_likelihood = mean_log_likelihood(
            self.policy,
            self.old_log_std,
            self.action_dim,
            self.observations,
            self.actions,
        )

        self.assertEqual(mean.shape, (self.batch_size, self.action_dim))
        self.assertEqual(log_likelihood.shape, (self.batch_size,))


if __name__ == "__main__":
    unittest.main()
