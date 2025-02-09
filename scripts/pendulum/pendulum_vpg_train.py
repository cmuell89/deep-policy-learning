import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork
from lr_challenge.learning.VPG import VanillaPolicyGradient
from lr_challenge.util import save_training_plots, NumpyEncoder
import datetime
import pprint
import os
import json
from collections import defaultdict


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"./data/pendulum/{timestamp}"
plots_dir = f"{run_dir}/plots"
video_dir = f"{run_dir}/videos"

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Initialize environment with video recording
env = gym.make("Pendulum-v1", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=video_dir,
    episode_trigger=lambda x: x % 100 == 0,
    video_length=1000,
)


# Create directories
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Create networks
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dims = [32, 32]


policy = GaussianActorPolicy.from_gym_env(
    env,
    device="cuda:0",
    hidden_dims=hidden_dims,
    activation=torch.nn.Tanh(),  # Tanh good for control tasks
    seed=SEED,
)

policy.log_stds.data.fill_(0)  # exp(0) ≈ 1 standard deviation

# Value network with orthogonal initialization (already implemented in ValueNetwork)
value_net = ValueNetwork.from_gym_env(
    env,
    device="cuda:0",
    hidden_dims=hidden_dims,
    seed=SEED,
)

# Initialize VPG agent
vpg = VanillaPolicyGradient(
    policy=policy,
    value_network=value_net,
    action_dim=action_dim,
    gamma=0.99,
    learning_rate=3e-3,
    device="cuda:0",
    seed=SEED,
)

# Modify training parameters
num_episodes = 500
max_steps = 200
learning_rate = 3e-3

# Training loop
returns = []
stats_history = defaultdict(list)
episode_lengths = []

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
        action, _ = policy.get_action(observation)
        action_np = action.detach().cpu().numpy()
        next_observation, reward, done, truncated, _ = env.step(action_np)

        observations.append(observation)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(float(truncated))

        observation = next_observation
        episode_reward += reward

        if done or truncated:
            break

    # Convert to tensors
    observations = torch.FloatTensor(np.array(observations)).to(vpg.device)
    actions = torch.FloatTensor(np.array(actions)).to(vpg.device)
    rewards = torch.FloatTensor(rewards).to(vpg.device)
    dones = torch.FloatTensor(dones).to(vpg.device)

    # Update policy and track stats
    stats = vpg.update(observations, actions, rewards, dones)
    returns.append(episode_reward)
    episode_lengths.append(len(rewards))

    for key, value in stats.items():
        stats_history[key].append(value)

    if (episode + 1) % 10 == 0:
        recent_avg = np.mean(returns[-10:])
        print(f"Episode {episode + 1}, Average Return: {recent_avg:.2f}")
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

# Save training data and plots
training_data = {
    "returns": returns,
    "episode_lengths": episode_lengths,
    "stats_history": {k: v for k, v in stats_history.items()},
    "config": {
        "hidden_dims": hidden_dims,
        "learning_rate": vpg.learning_rate,
        "gamma": vpg.gamma,
        "gae_lambda": vpg.gae_lambda,
        "max_steps": max_steps,
        "num_episodes": num_episodes,
        "seed": SEED,
    },
}

# Save training data
with open(f"{run_dir}/training_data.json", "w") as f:
    json.dump(training_data, f, indent=4, cls=NumpyEncoder)

# Generate and save plots
save_training_plots(returns, episode_lengths, stats_history, plots_dir)


# Helper class for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
