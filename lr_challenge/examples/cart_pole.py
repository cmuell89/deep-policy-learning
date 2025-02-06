import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork
from lr_challenge.learning.vpg import VanillaPolicyGradient
from lr_challenge.util import save_training_plots
import datetime
import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json


# Helper class for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


# Create run directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"./data/cartpole/{timestamp}"
plots_dir = f"{run_dir}/plots"
video_dir = f"{run_dir}/videos"

# Create directories
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Initialize environment with video recording
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=video_dir,
    episode_trigger=lambda x: x % 100 == 0,
    video_length=1000,
)
device = torch.device("cuda:0")
# Create networks
obs_dim = env.observation_space.shape[0]  # 4 for CartPole
action_dim = env.action_space.n  # 2 for CartPole (discrete)
hidden_dims = [64, 64]

# Create a dummy continuous action space for the policy
# This is a temporary workaround until we implement a proper discrete policy
dummy_action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(1,),
    dtype=np.float32,
)
dummy_env = type(
    "DummyEnv",
    (),
    {"observation_space": env.observation_space, "action_space": dummy_action_space},
)()

# Initialize policy
policy = GaussianActorPolicy.from_gym_env(
    dummy_env,
    device="cuda:0",
    hidden_dims=hidden_dims,
    activation=torch.nn.Tanh(),
    seed=SEED,
)

# Value network
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
    action_dim=1,  # Single continuous action
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=1e-3,
    device="cuda:0",
    seed=SEED,
)

# Training parameters
num_episodes = 600
max_steps = 500  # CartPole-v1 has 500 max steps
window_size = 20  # Larger window for smoother averaging
min_episodes = 100  # Minimum episodes before early stopping
patience = 250  # More patience
improvement_threshold = 20  # Minimum improvement required

# Training loop
returns = []
stats_history = defaultdict(list)
episode_lengths = []
best_avg_return = -float("inf")
episodes_without_improvement = 0

for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    observations = []
    actions = []
    rewards = []
    dones = []

    for step in range(max_steps):
        # Get continuous action and convert to discrete
        action_cont, _ = policy.get_action(observation)
        action_discrete = 1 if action_cont.item() > 0 else 0

        next_observation, reward, done, truncated, _ = env.step(action_discrete)

        # Store continuous action for training
        observations.append(observation)
        actions.append(action_cont.detach().cpu().numpy())
        rewards.append(reward)
        dones.append(float(done or truncated))

        observation = next_observation
        episode_reward += reward

        if done or truncated:
            break

    # Convert lists to tensors before update
    observations_tensor = torch.FloatTensor(np.array(observations)).to(device)
    actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)

    # Update stats tracking
    stats = vpg.update(
        observations_tensor, actions_tensor, rewards_tensor, dones_tensor
    )

    returns.append(episode_reward)
    episode_lengths.append(len(rewards))

    for key, value in stats.items():
        stats_history[key].append(value)

    # Early stopping logic
    if len(returns) >= window_size and episode >= min_episodes:
        current_avg_return = np.mean(returns[-window_size:])

        if current_avg_return > best_avg_return + 10:
            best_avg_return = current_avg_return
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1

        if episodes_without_improvement >= patience:
            print(f"\nStopping early at episode {episode + 1}")
            print(f"Best average return: {best_avg_return:.2f}")
            print(f"Current average return: {current_avg_return:.2f}")
            print(f"No significant improvement for {patience} episodes")
            break

    if (episode + 1) % 10 == 0:
        recent_avg = (
            np.mean(returns[-window_size:])
            if len(returns) >= window_size
            else np.mean(returns)
        )
        print(f"Episode {episode + 1}, Average Return: {recent_avg:.2f}")
        pprint.pprint(stats, indent=4, depth=4)

# Final test episode
print("\nTesting trained policy...")
state, _ = env.reset()
total_reward = 0
done = False

while not done:
    action_probs, _ = policy.get_action(state)
    action = torch.distributions.Categorical(logits=action_probs).sample()
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
