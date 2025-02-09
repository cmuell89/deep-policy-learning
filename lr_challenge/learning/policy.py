import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Union, Any
from lr_challenge.learning.functions import mean_log_likelihood


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
        activation: nn.Module = nn.ReLU(),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        layers: List[nn.Module] = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), activation])
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
        bounded_actions: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super(GaussianActorModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bounded_actions = bounded_actions
        self.device = device

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            # Initialize weights using Xavier/Glorot initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, activation()])
            prev_dim = hidden_dim

        # Final layer with smaller initialization
        final_layer = nn.Linear(prev_dim, output_dim)
        nn.init.uniform_(final_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.net = nn.Sequential(*layers)

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
        try:
            if isinstance(input, np.ndarray):
                input = torch.FloatTensor(input).to(self.device)

            # Input normalization
            x = (input - self.in_bias) / self.in_scale
            x = self.net(x)

            # For bounded actions, apply tanh and scale
            if hasattr(self, "out_scale") and self.out_scale is not None:
                x = torch.tanh(x)  # Squash to [-1, 1]
                action = x * self.out_scale + self.out_bias
            else:
                action = x  # Unbounded case - no squashing needed

            # Check for NaN values
            if torch.isnan(action).any():
                raise ValueError("NaN values detected in action output")
            return action

        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Input shape: {input.shape}")
            print(f"Input type: {type(input)}")
            raise e


class GaussianActorPolicy:
    """
    Gaussian policy for continuous action spaces with learnable mean and standard deviation.

    Attributes:
        hidden_dims: List of hidden layer dimensions
        activation: Activation function for hidden layers
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
        activation: nn.Module,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.hidden_dims = hidden_dims if hidden_dims else [256, 256]
        self.activation = activation if activation else nn.Tanh()
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
            torch.ones(action_dim, device=self.device) * -0.5, requires_grad=True
        )
        self.old_log_stds = nn.Parameter(
            torch.ones(action_dim, device=self.device) * -0.5,
            requires_grad=False,  # old parameters don't need gradients
        )

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
                self.activation,
                seed=self.seed,
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        if old_model is None:
            self.old_model = GaussianActorModel(
                self.obs_dim,
                self.action_dim,
                self.hidden_dims,
                self.activation,
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
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        mean = self.model(observation)
        std = torch.exp(self.log_stds)

        # Handle both single and multi-dimensional cases
        if mean.shape[-1] == 1:
            # 1D - use Normal
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
        else:
            # >1D - use MultivariateNormal
            batch_shape = mean.shape[:-1]

            covariance = torch.zeros(
                *batch_shape, self.action_dim, self.action_dim, device=self.device
            )
            covariance.diagonal(dim1=-2, dim2=-1).copy_(std.pow(2))

            normal = torch.distributions.MultivariateNormal(
                mean, scale_tril=torch.linalg.cholesky(covariance)
            )
            x_t = normal.rsample()

        # Apply tanh squashing
        y_t = torch.tanh(x_t)

        # Scale to action space
        action = y_t * self.model.out_scale + self.model.out_bias

        # Compute log prob (works for both Normal and MultivariateNormal)
        log_prob = normal.log_prob(x_t)

        # Apply tanh squashing correction
        squashing_correction = torch.sum(torch.log(1 - y_t.pow(2) + 1e-6), dim=-1)
        log_prob -= squashing_correction

        return action, {"mean": mean, "log_std": self.log_stds, "log_prob": log_prob}

    def parameters(self) -> Any:
        """Return an iterator over the policy's trainable parameters."""
        yield from self.model.parameters()
        yield self.log_stds

    def old_stats(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute statistics using the old policy network."""
        # Check if this is a bounded policy with valid scaling parameters

        mean, log_likelihood = mean_log_likelihood(
            self.old_model,
            self.old_log_stds,
            self.action_dim,
            observations,
            actions,
            bounded=self.old_model.bounded_actions,
        )
        return mean.clone(), self.old_log_stds.clone(), log_likelihood.clone()

    def stats(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute statistics using the current policy network."""
        # Check if this is a bounded policy with valid scaling parameters

        mean, log_likelihood = mean_log_likelihood(
            self.model,
            self.log_stds,
            self.action_dim,
            observations,
            actions,
            bounded=self.model.bounded_actions,
        )
        return mean.clone(), self.log_stds.clone(), log_likelihood.clone()

    def update_old_stats(self) -> None:
        """
        Update the old policy's statistics with the current policy's detached values.

        Args:
            mean: Mean of the current policy's action distribution
            log_stds: Log standard deviations of the current policy
            log_probs: Log probabilities of the current policy
        """
        self.old_model.load_state_dict(self.model.state_dict())

        with torch.no_grad():
            self.old_log_stds.copy_(self.log_stds)

    @staticmethod
    def from_gym_env(env, device, hidden_dims, activation, seed=None):
        """Generate a probabilistic policy network for the given environment"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Handle both Dict and Box observation spaces
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = env.observation_space["observation"].shape[0]
            obs_space = env.observation_space["observation"]
        else:
            obs_dim = env.observation_space.shape[0]
            obs_space = env.observation_space

        action_dim = env.action_space.shape[0]

        # Determine if environment has bounded actions
        is_bounded = (
            isinstance(env.action_space, gym.spaces.Box)
            and np.all(np.isfinite(env.action_space.low))
            and np.all(np.isfinite(env.action_space.high))
            and not (
                np.all(env.action_space.low == -np.inf)
                or np.all(env.action_space.high == np.inf)
            )
        )

        # Only create scaling parameters for bounded environments
        if is_bounded:
            out_scale = torch.FloatTensor(
                (env.action_space.high - env.action_space.low) / 2.0
            ).to(device)
            out_bias = torch.FloatTensor(
                (env.action_space.high + env.action_space.low) / 2.0
            ).to(device)
        else:
            out_scale = None
            out_bias = None

        # Handle unbounded spaces for observation normalization
        in_scale = torch.ones(obs_dim, device=device)
        in_bias = torch.zeros(obs_dim, device=device)

        # Only use high/low if they're finite
        if hasattr(obs_space, "high") and hasattr(obs_space, "low"):
            for i in range(obs_dim):
                if np.isfinite(obs_space.high[i]) and np.isfinite(obs_space.low[i]):
                    in_scale[i] = float(obs_space.high[i] - obs_space.low[i]) / 2.0
                    in_bias[i] = float(obs_space.high[i] + obs_space.low[i]) / 2.0

        # Create policy with seed
        model = GaussianActorModel(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            seed=seed,
            device=device,
            bounded_actions=is_bounded,
        )
        old_model = GaussianActorModel(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            seed=seed,
            device=device,
            bounded_actions=is_bounded,
        )

        # Only set output transformations if bounded
        if is_bounded:
            model.set_transformations(
                in_scale=in_scale,
                in_bias=in_bias,
                out_scale=out_scale,
                out_bias=out_bias,
            )
            old_model.set_transformations(
                in_scale=in_scale,
                in_bias=in_bias,
                out_scale=out_scale,
                out_bias=out_bias,
            )
        else:
            # For unbounded, only set input normalization
            model.set_transformations(in_scale=in_scale, in_bias=in_bias)
            old_model.set_transformations(in_scale=in_scale, in_bias=in_bias)

        policy = GaussianActorPolicy(
            action_dim,
            obs_dim,
            hidden_dims,
            activation,
            seed=seed,
            device=device,
        )
        policy._set_models(model, old_model)
        return policy
