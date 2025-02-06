import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork
from lr_challenge.learning.vpg import VanillaPolicyGradient
import datetime
import pprint
# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Initialize environment with video recording
env = gym.make("Pendulum-v1", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=f"./videos/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/",  # Will create this directory with timestamp
    episode_trigger=lambda x: x % 500 == 0,
)  # Record every 100th episode

# Create networks
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dims = [32, 32]

# Initialize policy with smaller std for more precise initial actions
policy = GaussianActorPolicy.from_gym_env(
    env,
    device="cuda:0",
    hidden_dims=hidden_dims,
    activation=torch.nn.Tanh(),  # Tanh good for control tasks
    seed=SEED,
)
# Initialize log_stds to smaller value for more precise initial actions
with torch.no_grad():
    policy.log_stds.data.fill_(-1.0)  # exp(0) ≈ 1 standard deviation

# Value network with orthogonal initialization (already implemented in ValueNetwork)
value_net = ValueNetwork.from_gym_env(
    env,
    device="cuda:0",
    hidden_dims=hidden_dims,  # Same architecture as policy
    seed=SEED,
)

# Initialize VPG agent
vpg = VanillaPolicyGradient(
    policy=policy,
    value_network=value_net,
    action_dim=action_dim,
    gamma=0.99,  # Standard discount
    gae_lambda=0.95,  # GAE lambda
    learning_rate=3e-4,  # Standard LR for Adam
    device="cuda:0",
    seed=SEED,
)

# Modify training parameters
num_episodes = 4000  # More episodes
max_steps = 100
learning_rate = 3e-4  # Standard LR for Adam

# Initialize optimizers with better parameters
vpg.optimizer_policy = torch.optim.Adam(policy.parameters(), lr=learning_rate)
vpg.optimizer_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)

# Training loop
returns = []

for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []

    # Collect trajectory
    for step in range(max_steps):
        # Get action from policy
        action, _ = policy.get_action(observation)
        # Convert action tensor to numpy, detaching from computation graph
        action_np = action.detach().cpu().numpy()
        next_observation, reward, done, truncated, _ = env.step(action_np)
        # Store transition
        observations.append(observation)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(float(truncated))

        observation = next_observation
        episode_reward += reward

        if done or truncated:
            break

    # Convert to tensors
    observations = torch.FloatTensor(np.array(observations))
    next_observations = torch.FloatTensor(np.array(next_observations))
    actions = torch.FloatTensor(np.array(actions))
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    # Update policy
    stats = vpg.update(observations, actions, rewards, dones)
    returns.append(episode_reward)

    # Print progress
    if (episode + 1) % 10 == 0:
        avg_return = np.mean(returns[-10:])
        print(f"Episode {episode + 1}, Average Return: {avg_return:.2f}")
        pprint.pprint(stats, indent=4, depth=4)

# Final test episode will also be recorded
print("\nTesting trained policy...")
state, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, _ = policy.get_action(state)
    action_np = action.detach().cpu().numpy()
    state, reward, done, truncated, _ = env.step(action_np)
    total_reward += reward

    if done or truncated:
        break

print(f"Test episode reward: {total_reward:.2f}")
env.close()
