import unittest
import torch
import numpy as np
import gymnasium as gym
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork


class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Pendulum-v1")
        self.device = "cpu"

    def test_gaussian_policy_initialization(self):
        hidden_dims = [32, 32]
        activation = torch.nn.Tanh()

        policy = GaussianActorPolicy.from_gym_env(
            self.env, self.device, hidden_dims, activation, seed=42
        )

        # Test basic attributes
        self.assertEqual(policy.action_dim, self.env.action_space.shape[0])
        self.assertEqual(policy.obs_dim, self.env.observation_space.shape[0])
        self.assertEqual(policy.hidden_dims, hidden_dims)
        self.assertIsInstance(policy.nonlinearity, torch.nn.Tanh)

        # Test log_stds initialization
        self.assertEqual(policy.log_stds.shape, (self.env.action_space.shape[0],))
        self.assertTrue(
            torch.allclose(policy.log_stds, torch.zeros_like(policy.log_stds))
        )
        self.assertTrue(policy.log_stds.requires_grad)

    def test_gaussian_policy_forward(self):
        policy = GaussianActorPolicy.from_gym_env(
            self.env, self.device, [32, 32], torch.nn.Tanh(), seed=42
        )

        # Test single observation
        obs = self.env.observation_space.sample()
        action, info = policy.get_action(obs)

        self.assertEqual(action.shape, self.env.action_space.shape)
        self.assertIn("mean", info)
        self.assertIn("log_std", info)
        self.assertEqual(info["mean"].shape, action.shape)
        self.assertEqual(info["log_std"].shape, (self.env.action_space.shape[0],))

        # Test batch of observations
        batch_size = 10
        obs_batch = np.stack(
            [self.env.observation_space.sample() for _ in range(batch_size)]
        )
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        means = policy(obs_tensor)

        self.assertEqual(means.shape, (batch_size,) + self.env.action_space.shape)

    def test_value_network_initialization(self):
        hidden_dims = [32, 32]
        value_net = ValueNetwork.from_gym_env(self.env, self.device, hidden_dims, seed=42)

        # Test network structure
        self.assertIsInstance(value_net, torch.nn.Module)
        self.assertTrue(len(list(value_net.parameters())) > 0)

        # Test output shape
        obs = self.env.observation_space.sample()
        value = value_net(obs)
        self.assertEqual(value.shape, (1,))

        # Test batch processing
        batch_size = 10
        obs_batch = np.stack([self.env.observation_space.sample() for _ in range(batch_size)])
        values = value_net(obs_batch)
        self.assertEqual(values.shape, (batch_size, 1))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_policy_device_transfer(self):
        device = "cuda:0"
        policy = GaussianActorPolicy.from_gym_env(
            self.env, device, [32, 32], torch.nn.Tanh(), seed=42
        )

        # Check if model and parameters are on correct device
        self.assertEqual(next(policy.model.parameters()).device.type, "cuda")
        self.assertEqual(policy.log_stds.device.type, "cuda")

        # Test forward pass on GPU
        obs = torch.FloatTensor(self.env.observation_space.sample()).to(device)
        action, info = policy.get_action(obs)
        self.assertEqual(action.device.type, "cuda")
        self.assertEqual(info["mean"].device.type, "cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_value_network_device_transfer(self):
        device = "cuda:0"
        value_net = ValueNetwork.from_gym_env(self.env, device, [32, 32], seed=42)

        # Check if network is on correct device
        self.assertEqual(next(value_net.parameters()).device.type, "cuda")

        # Test forward pass on GPU
        obs = torch.FloatTensor(self.env.observation_space.sample()).to(device)
        value = value_net(obs)
        self.assertEqual(value.device.type, "cuda")

    def test_policy_deterministic_with_seed(self):
        """Test that same seed produces same outputs"""
        seed = 42
        policy1 = GaussianActorPolicy.from_gym_env(
            self.env, self.device, [32, 32], torch.nn.Tanh(), seed=seed
        )
        policy2 = GaussianActorPolicy.from_gym_env(
            self.env, self.device, [32, 32], torch.nn.Tanh(), seed=seed
        )

        obs = self.env.observation_space.sample()
        torch.manual_seed(seed)  # For random action noise
        action1, _ = policy1.get_action(obs)
        torch.manual_seed(seed)  # Reset seed
        action2, _ = policy2.get_action(obs)

        self.assertTrue(torch.allclose(action1, action2))


if __name__ == '__main__':
    unittest.main()
