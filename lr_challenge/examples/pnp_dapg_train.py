import gymnasium as gym
import panda_gym  # noqa
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import time
from lr_challenge.learning.policy import (
    generate_probabilistic_policy,
    generate_value_network,
)
from lr_challenge.learning.policy_gradient import DAPG
import os

# Set global seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device consistently
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Update deprecated tensor type setting
torch.set_default_dtype(torch.float32)
torch.set_default_device(device)

# Initialize environment
env = gym.make(
    "PandaSlideDense",
    render_mode="human",
    renderer="OpenGL",
    render_width=1080,
    render_height=1080,
)

# Create networks with seeds
policy_net = generate_probabilistic_policy(env, seed=SEED).to(device)
value_net = generate_value_network(env, seed=SEED).to(device)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # Use Xavier/Glorot initialization with a smaller gain for more stability
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # Initialize biases to small values
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


policy_net.apply(init_weights)
value_net.apply(init_weights)

# Perform a dummy forward pass to ensure initialization
with torch.no_grad():
    state, _ = env.reset()
    obs = torch.tensor(state["observation"], dtype=torch.float32, device=device)
    # Test policy network - Move to CPU before converting to numpy
    obs_cpu = obs.cpu()
    action, info = policy_net.get_action(obs_cpu.numpy())
    # Test value network
    _ = value_net(obs)

    # Initialize action std to a conservative value
    if hasattr(policy_net, "log_std"):
        policy_net.log_std.data.fill_(-1.0)  # This corresponds to std ≈ 0.37

print("Networks initialized with stable parameters")

# Initialize DAPG with seed
dapg = DAPG(
    policy_network=policy_net,
    value_network=value_net,
    action_dim=env.action_space.shape[0],
    seed=SEED,
    learning_rate=1e-3,  # vf_learn_rate: 1e-3
    gamma=0.995,  # rl_gamma: 0.995
    delta=0.05,  # rl_step_size: 0.05
    gae_lambda=0.97,  # rl_gae: 0.97
    lam_0=1e-2,  # lam_0: 1e-2
    lam_1=0.95,  # lam_1: 0.95
    damping_coeff=0.01,
    iter_count=0,
    hvp_subsample=0.1,
)

# Training parameters
trajectories_per_update = 15  # rl_num_traj: 200
num_episodes = 10000  # rl_num_iter: 100
max_steps_per_episode = 1000  # vf_epmaxochs: 2

# Initialize tracking variables
returns_history = []
value_losses = []
kl_divs = []
running_return = deque(maxlen=10)  # For moving average

# Training loop
trajectories = []
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    trajectory = {"observations": [], "actions": [], "rewards": [], "dones": []}

    for step in range(max_steps_per_episode):
        # Get action from policy - ensure observation is on CPU
        obs = torch.tensor(state["observation"], dtype=torch.float32, device=device)
        action, _ = dapg.policy.get_action(obs.cpu().numpy())

        # Take action in environment
        next_state, reward, done, truncated, info = env.step(action)

        # Store in trajectory
        trajectory["observations"].append(state["observation"])
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["dones"].append(done)

        state = next_state
        episode_reward += reward

        if done or truncated:
            break

    # Convert trajectory lists to numpy arrays
    for k in trajectory:
        trajectory[k] = np.array(trajectory[k])

    trajectories.append(trajectory)

    # Update policy after collecting multiple trajectories
    if len(trajectories) >= trajectories_per_update:
        stats = dapg.update(trajectories)
        trajectories = []  # Reset collection after update

        if stats:
            print("\nTraining Stats:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            if "value_loss" in stats:
                value_losses.append(stats["value_loss"])
            if "kl_div" in stats:
                kl_divs.append(stats["kl_div"])
    running_return.append(episode_reward)

    # Print episode statistics
    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

    # Periodically test the policy
    if (episode + 1) % 100 == 0:
        print(f"\nTesting policy at episode {episode + 1}")
        print(f"Moving Avg Return: {np.mean(running_return):.2f}")

        # Run test episode using existing environment
        test_state, _ = env.reset()
        test_reward = 0
        done = False

        while not done:
            action, _ = policy_net.get_action(test_state["observation"])
            test_state, reward, done, truncated, _ = env.step(action)
            test_reward += reward
            time.sleep(0.15)  # Slow down rendering

            if done or truncated:
                break

        print(f"Test Episode Reward: {test_reward:.2f}")
        print("------------------------")

# Create models directory if it doesn't exist
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(models_dir, exist_ok=True)

# Save model using absolute path with timestamp
model_path = os.path.join(
    models_dir, f"pnp_model_{int(time.time())}"
)  # Unique timestamp identifier
dapg.save_model(model_path)


matplotlib.use("Agg")  # Set the backend to non-interactive

# Plot training results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(returns_history)
plt.title("Returns over Training")
plt.xlabel("Episode")
plt.ylabel("Return")

if value_losses:
    plt.subplot(132)
    plt.plot(value_losses)
    plt.title("Value Loss over Training")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

if kl_divs:
    plt.subplot(133)
    plt.plot(kl_divs)
    plt.title("KL Divergence over Training")
    plt.xlabel("Episode")
    plt.ylabel("KL Div")

plt.tight_layout()
plt.savefig("training_plots.png")
plt.close()

env.close()

# Final policy test
print("\nFinal Policy Test:")
test_env = gym.make(
    "PandaPickAndPlaceDense-v3",
    render_mode="human",
    renderer="OpenGL",
    render_width=1080,
    render_height=1080,
)

state, _ = test_env.reset()
total_reward = 0
done = False

while not done:
    action, _ = policy_net.get_action(state["observation"])
    state, reward, done, truncated, _ = test_env.step(action)
    total_reward += reward
    time.sleep(0.01)

    if done or truncated:
        break

print(f"Final test episode reward: {total_reward:.2f}")
test_env.close()
