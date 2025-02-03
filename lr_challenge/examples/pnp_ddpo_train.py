import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create and wrap the environment
def make_env():
    return gym.make("PandaPickAndPlaceDense-v3", render_mode="human")

env = DummyVecEnv([make_env])

# Get action space dimensions for noise
n_actions = env.action_space.shape[-1]

# Configure action noise
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)  # 10% of action range for exploration
)

# Create DDPG model with tuned hyperparameters
model = DDPG(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-4,          # Slightly lower learning rate for stability
    buffer_size=1_000_000,       # Large buffer for better sampling
    learning_starts=1000,        # Collect some experience before learning
    batch_size=256,             # Larger batch size for better estimates
    tau=0.005,                  # Soft update coefficient
    gamma=0.98,                 # Discount factor
    train_freq=(1, "episode"),  # Update policy every episode
    gradient_steps=-1,          # Use all steps in buffer
    action_noise=action_noise,  # Add exploration noise
    policy_kwargs=dict(
        net_arch=dict(
            pi=[400, 300],  # Actor architecture
            qf=[400, 300]   # Critic architecture
        )
    ),
    verbose=1
)

# Train the model
model.learn(
    total_timesteps=100_000,
    log_interval=10
)

# Save the trained model
model.save("panda_ddpg_trained")

# Run environment
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)  # No noise during testing
    obs, rewards, dones, infos = env.step(action)
    
    if dones[0]:
        if "terminal_observation" in infos[0]:
            terminal_obs = infos[0]["terminal_observation"]