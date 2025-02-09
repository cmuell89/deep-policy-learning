import gymnasium as gym
import panda_gym  # noqa
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork
from lr_challenge.algorithms.PPO import ProximalPolicyOptimization
from lr_challenge.util import save_training_plots
import datetime
import pprint
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
run_dir = f"./data/panda_ppo/{timestamp}"
plots_dir = f"{run_dir}/plots"
video_dir = f"{run_dir}/videos"

# Create directories
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device consistently
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize environment
env = gym.make(
    "PandaPickAndPlaceDense-v3",
    renderer="OpenGL",
    render_mode="human",
    render_width=720,
    render_height=720,
)
# Initialize networks
hidden_dims = [256, 256]  # Larger network for more complex task
policy = GaussianActorPolicy.from_gym_env(
    env,
    device=device,
    hidden_dims=hidden_dims,
    activation=torch.nn.ReLU(),  # ReLU often works better for robotics
    seed=SEED,
)

value_net = ValueNetwork.from_gym_env(
    env,
    device=device,
    hidden_dims=hidden_dims,
    seed=SEED,
)

# Initialize PPO agent with robotics-tuned hyperparameters
ppo = ProximalPolicyOptimization(
    policy=policy,
    value_network=value_net,
    action_dim=env.action_space.shape[0],
    gamma=0.99,
    gae_lambda=0.95,
    clipping_epsilon=0.2,
    learning_rate=3e-4,
    n_epochs=5,
    batch_size=256,
    ent_coef=0.01,
    vf_coef=0.7,
    device=device,
    seed=SEED,
)


# Training parameters
num_episodes = 250  # More episodes for complex task
trajectories_per_update = 5  # More steps per update
max_steps = 2000  # Longer episodes
window_size = 20

# Training loop
returns = []
stats_history = defaultdict(list)
episode_lengths = []

for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    observations = []
    actions = []
    rewards = []
    dones = []
    trajectory_count = 0

    while trajectory_count < trajectories_per_update:
        observation, _ = env.reset()
        curr_obs_traj = []
        curr_act_traj = []
        curr_rew_traj = []
        curr_done_traj = []

        for _ in range(max_steps):
            # Get continuous action directly (no need for discrete conversion)
            action, _ = policy.get_action(observation["observation"])
            action_numpy = action.detach().cpu().numpy()
            next_observation, reward, done, truncated, _ = env.step(action_numpy)

            # Store trajectory data
            curr_obs_traj.append(
                observation["observation"]
            )  # Note: using observation dict
            curr_act_traj.append(action_numpy)
            curr_rew_traj.append(reward)
            curr_done_traj.append(float(done or truncated))

            observation = next_observation
            episode_reward += reward

        trajectory_count += 1

        observations.extend(curr_obs_traj)
        actions.extend(curr_act_traj)
        rewards.extend(curr_rew_traj)
        dones.extend(curr_done_traj)

    # Convert lists to tensors before update
    observations_tensor = torch.FloatTensor(np.array(observations)).to(device)
    actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
    rewards_tensor = torch.FloatTensor(np.array(rewards)).to(device)
    dones_tensor = torch.FloatTensor(np.array(dones)).to(device)

    # Update using PPO
    stats = ppo.update(
        observations_tensor, actions_tensor, rewards_tensor, dones_tensor
    )

    returns.append(episode_reward)
    episode_lengths.append(len(rewards))

    for key, value in stats.items():
        stats_history[key].append(value)

    recent_avg = (
        np.mean(returns[-window_size:])
        if len(returns) >= window_size
        else np.mean(returns)
    )
    print(f"Episode {episode}, Average Return: {recent_avg:.2f}")
    pprint.pprint(stats, indent=4, depth=4)

    # if episode % 100 == 0:
    #     env.reset()
    #     observation, _ = env.reset()
        
    #     frame = env.render()
    #     height, width, layers = frame.shape
    #     steps = 0
        
    #     # Create a unique video filename for each recording
    #     video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
    #     video = cv2.VideoWriter(
    #         video_path,
    #         cv2.VideoWriter_fourcc(*'mp4v'),
    #         60,
    #         (width, height)
    #     )

    #     while steps < 1000:
    #         action, _ = policy.get_action(observation["observation"])
    #         observation, reward, done, truncated, _ = env.step(action.detach().cpu().numpy())
    #         steps += 1

    #         frame = env.render()
    #         # Convert frame from RGB to BGR for OpenCV
    #         video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    #         if done or truncated:
    #             break

    #     video.release()


n_test_episodes = 10
episode_rewards = []

for episode in range(n_test_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _ = policy.get_action(state["observation"])
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

        if done or truncated:
            break

    episode_rewards.append(episode_reward)
    print(f"Test episode {episode + 1} reward: {episode_reward:.2f}")

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(
    f"\nAverage test reward over {n_test_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}"
)
print(f"Videos saved to: {video_dir}")

env.close()

# Save training data and plots
training_data = {
    "returns": returns,
    "episode_lengths": episode_lengths,
    "stats_history": {k: v for k, v in stats_history.items()},
    "config": {
        "hidden_dims": hidden_dims,
        "learning_rate": ppo.learning_rate,
        "gamma": ppo.gamma,
        "gae_lambda": ppo.gae_lambda,
        "clipping_epsilon": ppo.clipping_epsilon,
        "n_epochs": ppo.n_epochs,
        "batch_size": ppo.batch_size,
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
