import gymnasium as gym
import panda_gym
import torch
import time
import os
from lr_challenge.learning.policy import generate_probabilistic_policy
from lr_challenge.learning.policy_gradient import DAPG

# Set device consistently
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize environment
env = gym.make("PandaPickAndPlaceDense-v3",     
    render_mode="human",
    renderer="OpenGL",
    render_width=1080,
    render_height=1080,
)

# Create policy network with same architecture
policy_net = generate_probabilistic_policy(env).to(device)

# Load the saved model
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
# You can specify the timestamp of the model you want to load
timestamp = "YYYYMMDD_HHMMSS"  # Replace with the actual timestamp of the model you want to load
model_path = os.path.join(models_dir, f"pnp_dapg_{timestamp}")

# Load the model state
policy_net.load_state_dict(torch.load(model_path))
policy_net.eval()  # Set to evaluation mode

# Run the environment with loaded model
state, _ = env.reset()
total_reward = 0
done = False

print("\nRunning saved model...")
while not done:
    # Get action from loaded policy
    with torch.no_grad():
        action, _ = policy_net.get_action(state["observation"])
    
    # Take action in environment
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    # Optional: slow down visualization
    time.sleep(0.01)
    
    if done or truncated:
        break

print(f"Episode finished with total reward: {total_reward:.2f}")
env.close()