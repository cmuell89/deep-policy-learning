import gymnasium as gym
import panda_gym  # noqa
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from lr_challenge.algorithms.PPO import ProximalPolicyOptimization
from lr_challenge.learning.policy import GaussianActorPolicy, ValueNetwork
from lr_challenge.util import NumpyEncoder, VideoRecorderWrapper


def main():
    # Configuration
    config = {
        # Environment
        "env_id": "PandaSlideDense",
        "seed": 42,
        # Architecture
        "hidden_dims": [256, 256],
        "activation": "ReLU",
        # PPO specific parameters
        "num_episodes": 200,
        "max_steps": 1000,
        "learning_rate": 1e-3,
        "trajectories_per_episode": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "n_epochs": 5,
        "batch_size": 128,
        "ent_coef": 0.02,
        "vf_coef": 0.9,
        # Hardware
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['env_id']}_ppo_{timestamp}"
    exp_dir = Path("./runs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, cls=NumpyEncoder)

    # Set seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create environments
    train_env = gym.make(config["env_id"], render_mode="rgb_array")
    video_env = gym.make(config["env_id"], render_mode="rgb_array")
    video_env = VideoRecorderWrapper(video_env, video_dir=exp_dir / "videos")

    train_env.reset(seed=config["seed"])
    video_env.reset(seed=config["seed"])

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

    # Initialize PPO algorithm
    ppo = ProximalPolicyOptimization(
        policy=policy,
        value_network=value_net,
        action_dim=train_env.action_space.shape[0],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clipping_epsilon=config["clip_epsilon"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_steps=config["max_steps"],
        num_episodes=config["num_episodes"],
        trajectories_per_episode=config["trajectories_per_episode"],
        device=config["device"],
        seed=config["seed"],
    )

    # Initialize tensorboard
    writer = SummaryWriter(exp_dir / "tensorboard")

    try:
        # Train
        print(f"Starting training... Experiment: {exp_name}")
        print(f"Tensorboard logs: {exp_dir}/tensorboard")
        training_info = ppo.train(env=train_env, writer=writer, video_env=video_env)

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
                "config": ppo.config,
            },
            exp_dir / "interrupted_checkpoint.pt",
        )

    finally:
        writer.close()
        train_env.close()
        print(f"\nExperiment data saved to: {exp_dir}")


if __name__ == "__main__":
    main()
