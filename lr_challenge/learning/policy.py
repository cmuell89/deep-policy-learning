import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np



def generate_probabilistic_policy(env):
    """Generate a probabilistic policy network for the given environment"""
    obs_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    policy = TanhNormalPolicy(obs_dim, action_dim)
    
    # Set action bounds using register_buffer
    policy.action_scale = torch.FloatTensor(
        (env.action_space.high - env.action_space.low) / 2.0
    )
    policy.action_bias = torch.FloatTensor(
        (env.action_space.high + env.action_space.low) / 2.0
    )
    
    return policy

def generate_value_network(env):
    """Generate a value network for the given environment"""
    obs_dim = env.observation_space["observation"].shape[0]
    
    class ValueNetwork(nn.Module):
        def __init__(self, obs_dim, hidden_dims=[256, 256]):
            super().__init__()
            
            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU()
                ])
                prev_dim = hidden_dim
                
            layers.append(nn.Linear(hidden_dims[-1], 1))
            
            self.net = nn.Sequential(*layers)
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
                
        def forward(self, obs):
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            return self.net(obs)
    
    return ValueNetwork(obs_dim)

class TanhNormalPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        # Build the policy network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        # Mean and log_std heads for the normal distribution
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Register buffers for action scaling (these will move to the correct device with the model)
        self.register_buffer('action_scale', torch.tensor(1.0))
        self.register_buffer('action_bias', torch.tensor(0.0))
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.action_scale.device)
        else:
            obs = obs.to(self.action_scale.device)
            
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Constrain log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        
        # Create normal distribution
        std = log_std.exp()
        dist = Normal(mean, std)
        
        # Sample action and apply tanh transformation
        x_t = dist.rsample()  # Use reparameterization trick
        action = torch.tanh(x_t)
        
        # Scale and shift the action to match the environment's action space
        action = action * self.action_scale + self.action_bias
        
        return action, {
            'mean': mean,
            'std': std,
            'log_std': log_std,
            'dist': dist,
            'log_prob': dist.log_prob(x_t).sum(-1)  # Sum across action dimensions
        }

    def get_action(self, obs):
        """Method to get action and additional info for training"""
        with torch.no_grad():
            action, info = self.forward(obs)
            return action.cpu().numpy(), {
                'mean': info['mean'].cpu().numpy(),
                'std': info['std'].cpu().numpy(),
                'log_prob': info['log_prob'].cpu().numpy()
            }
