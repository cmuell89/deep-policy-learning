import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from lr_challenge.algorithms.VPG import VanillaPolicyGradient
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork
from lr_challenge.util import NumpyEncoder, ContinuousToDiscreteWrapper, VideoRecorderWrapper

def main():
    # Configuration
    config = {
        # Environment
        "env_id": "CartPole-v1",
        "seed": 42,
        # Architecture
        "hidden_dims": [32, 32],
        "activation": "Tanh",
        # Training
        "num_episodes": 150,
        "max_steps": 500,
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "entropy_coef": 0.01,
        "max_grad_norm": 10.0,
        # Hardware
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['env_id']}_vpg_{timestamp}"
    exp_dir = Path("./runs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)

    # Set seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create environments
    train_env_discrete = gym.make(config["env_id"], render_mode="rgb_array")
    video_env_discrete = gym.make(config["env_id"], render_mode="rgb_array")

    # Create a continuous wrapper for the environments
    train_env = ContinuousToDiscreteWrapper(train_env_discrete)
    video_env = ContinuousToDiscreteWrapper(video_env_discrete)

    video_env = VideoRecorderWrapper(
        video_env,
        video_dir=exp_dir / "videos",
    )
    
    train_env.reset(seed=config["seed"])

    if config["activation"] == "Tanh":
        activation = torch.nn.Tanh
    elif config["activation"] == "ReLU":
        activation = torch.nn.ReLU
    else:
        raise ValueError(f"Activation function {config['activation']} not supported")

    # Initialize networks
    policy = GaussianActorPolicy.from_gym_env(
        env=train_env,
        device=config["device"],
        hidden_dims=config["hidden_dims"],
        activation=activation,
        seed=config["seed"],
    )

    value_net = ValueNetwork.from_gym_env(
        env=train_env,
        device=config["device"],
        hidden_dims=config["hidden_dims"],
        seed=config["seed"],
    )

    # Initialize algorithm
    vpg = VanillaPolicyGradient(
        policy=policy,
        value_network=value_net,
        action_dim=train_env.action_space.shape[0],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        max_steps=config["max_steps"],
        num_episodes=config["num_episodes"],
        device=config["device"],
    )

    # Initialize tensorboard
    writer = SummaryWriter(exp_dir / "tensorboard")

    try:
        # Train
        print(f"Starting training... Experiment: {exp_name}")
        print(f"Tensorboard logs: {exp_dir}/tensorboard")
        training_info = vpg.train(env=train_env, writer=writer, video_env=video_env)

        # Save final model
        torch.save(
            {
                "policy_state": policy.model.state_dict(),
                "value_state": value_net.state_dict(),
                "training_info": training_info,
                "config": config,
            },
            exp_dir / "final_model.pt",
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        torch.save(
            {
                "policy_state": policy.model.state_dict(),
                "value_state": value_net.state_dict(),
                "config": vpg.config,
            },
            exp_dir / "interrupted_checkpoint.pt",
        )

    finally:
        writer.close()
        train_env.close()
        video_env.close()
        print(f"\nExperiment data saved to: {exp_dir}")


if __name__ == "__main__":
    main()
