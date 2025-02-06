import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Union, Any
from lr_challenge.learning.functions import mean_log_likelihood
from torch.distributions import Normal


class ValueNetwork(nn.Module):
    """
    Neural network that estimates the value function for a given state.

    Attributes:
        net: Sequential neural network for value estimation
        obs_dim: Dimension of observation space
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [32, 32],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        layers: List[nn.Module] = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.Tanh()])
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the value network."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        return self.net(obs)

    @staticmethod
    def from_gym_env(
        env: gym.Env,
        device: Union[str, torch.device],
        hidden_dims: List[int] = [32, 32],
        seed: Optional[int] = None,
    ) -> "ValueNetwork":
        """
        Create a value network for a given gym environment.

        Args:
            env: Gymnasium environment
            device: Device to place network on
            hidden_dims: Dimensions of hidden layers
            seed: Random seed for initialization

        Returns:
            Initialized value network
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = env.observation_space["observation"].shape[0]
        else:
            obs_dim = env.observation_space.shape[0]
        return ValueNetwork(obs_dim, hidden_dims=hidden_dims, seed=seed).to(device)


class GaussianActorModel(nn.Module):
    """
    Neural network model for the Gaussian policy that outputs action means.

    Attributes:
        input_dim: Dimension of input (observation) space
        output_dim: Dimension of output (action) space
        activation: Activation function for hidden layers
        device: Device to place network on
        linear_layers: List of linear layers
        in_bias: Input normalization bias
        in_scale: Input normalization scale
        out_bias: Output normalization bias
        out_scale: Output normalization scale
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: nn.Module,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        super(GaussianActorModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.device = device

        layer_sizes = (input_dim,) + tuple(hidden_dims) + (output_dim,)
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def set_transformations(
        self,
        in_scale: Optional[torch.Tensor] = None,
        in_bias: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
        out_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Set input and output normalization parameters."""
        self.in_bias = (
            in_bias.to(self.device)
            if in_bias is not None
            else torch.zeros(self.input_dim).to(self.device)
        )
        self.in_scale = (
            in_scale.to(self.device)
            if in_scale is not None
            else torch.ones(self.input_dim).to(self.device)
        )
        self.out_bias = (
            out_bias.to(self.device)
            if out_bias is not None
            else torch.zeros(self.output_dim).to(self.device)
        )
        self.out_scale = (
            out_scale.to(self.device)
            if out_scale is not None
            else torch.ones(self.output_dim).to(self.device)
        )

    def forward(self, input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the actor network with input/output normalization."""
        if isinstance(input, np.ndarray):
            input = torch.FloatTensor(input).to(self.device)
        x = (input - self.in_bias) / self.in_scale
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        x = self.activation(x)
        action = x * self.out_scale + self.out_bias
        return action


class GaussianActorPolicy:
    """
    Gaussian policy for continuous action spaces with learnable mean and standard deviation.

    Attributes:
        hidden_dims: List of hidden layer dimensions
        nonlinearity: Activation function for hidden layers
        model: Current policy network
        old_model: Previous policy network for updates
        action_dim: Number of action dimensions
        obs_dim: Number of observation dimensions
        device: Device to run computations on
        log_stds: Learnable log standard deviations
        old_log_stds: Previous log standard deviations
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        hidden_dims: List[int],
        nonlinearity: nn.Module,
        device: Union[str, torch.device],
        seed: Optional[int] = None,
    ) -> None:
        self.hidden_dims = hidden_dims if hidden_dims else [256, 256]
        self.nonlinearity = nonlinearity if nonlinearity else nn.Tanh()
        self.model: Optional[GaussianActorModel] = None
        self.old_model: Optional[GaussianActorModel] = None
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.seed = seed
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._set_models()
        # Log Std are learnable parameters.
        self.log_stds = nn.Parameter(
            torch.zeros(action_dim, device=self.device), requires_grad=True
        )
        self.old_log_stds = torch.zeros_like(self.log_stds, device=self.device)

    def _set_models(
        self,
        model: Optional[GaussianActorModel] = None,
        old_model: Optional[GaussianActorModel] = None,
    ) -> None:
        """Initialize or update the current and old policy networks."""
        if model is None:
            self.model = GaussianActorModel(
                self.obs_dim,
                self.action_dim,
                self.hidden_dims,
                self.nonlinearity,
                seed=self.seed,
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        if old_model is None:
            self.old_model = GaussianActorModel(
                self.obs_dim,
                self.action_dim,
                self.hidden_dims,
                self.nonlinearity,
                seed=self.seed,
            ).to(self.device)
        else:
            self.old_model = old_model.to(self.device)

    def __call__(self, observations: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the current policy network."""
        return self.model(observations)

    def get_action(
        self, observation: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample an action from the policy given an observation."""
        mean = self.model(observation)
        std = torch.exp(self.log_stds)

        # Sample from normal distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Use reparameterization trick

        # Apply tanh squashing
        y_t = torch.tanh(x_t)

        # Scale to action space
        action = y_t * self.model.out_scale + self.model.out_bias

        # For computing log prob, need to account for tanh squashing
        log_prob = normal.log_prob(x_t)

        # Apply tanh squashing correction
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)

        return action, {"mean": mean, "log_std": self.log_stds, "log_prob": log_prob}

    def parameters(self) -> Any:
        """Return an iterator over the policy's trainable parameters."""
        yield from self.model.parameters()
        yield self.log_stds

    def old_stats(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute statistics using the old policy network."""
        mean, log_likelihood = mean_log_likelihood(
            self.old_model, self.log_std, self.action_dim, observations, actions
        )
        return mean, self.old_log_stds, log_likelihood

    def stats(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute statistics using the current policy network."""
        mean, log_likelihood = mean_log_likelihood(
            self.model, self.log_stds, self.action_dim, observations, actions
        )
        return mean, log_likelihood

    @staticmethod
    def from_gym_env(env, device, hidden_dims, activation, seed=None):
        """Generate a probabilistic policy network for the given environment"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Handle both Dict and Box observation spaces
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = env.observation_space["observation"].shape[0]
        else:
            obs_dim = env.observation_space.shape[0]

        action_dim = env.action_space.shape[0]

        # Determine action scale and action bias from environment
        in_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        in_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )
        policy = GaussianActorPolicy(
            action_dim,
            obs_dim,
            hidden_dims,
            activation,
            seed=seed,
            device=device,
        )
        # Create policy with seed
        model = GaussianActorModel(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            seed=seed,
            device=device,
        )
        old_model = GaussianActorModel(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            seed=seed,
            device=device,
        )
        model.set_transformations(in_scale, in_bias)
        old_model.set_transformations(in_scale, in_bias)
        policy._set_models(model, old_model)
        return policy
