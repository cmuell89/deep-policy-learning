import gymnasium as gym
from pathlib import Path
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
import torch
import os
from datetime import datetime


def make_env(rank: int, seed: int = 0):
    """
    Create a wrapped, monitored SubprocVecEnv for PandaSlide
    """
    def _init():
        env = gym.make(
            "PandaSlideDense",
            render_mode="rgb_array",
        )
        env.reset(seed=seed + rank)  # Seed each env differently
        return env
    
    set_random_seed(seed)
    return _init


def main():
    config = {
        # Environment
        "env_id": "PandaSlideDense",
        "total_timesteps": 1_000_000,
        "num_envs": 4,  # Number of parallel environments
        "seed": 42,
        # PPO parameters
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "normalize_advantage": True,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.05,
        # Hardware
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    # Create run directory for logs and models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['env_id']}_sb3_ppo_{timestamp}"
    exp_dir = Path("./runs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create parallel environments
    env = SubprocVecEnv([
        make_env(i, config["seed"]) 
        for i in range(config["num_envs"])
    ])

    # Create single evaluation environment
    eval_env = SubprocVecEnv([make_env(0, config["seed"] + 1000)])
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=exp_dir / "videos",
        record_video_trigger=lambda x: x % 200 == 0,
        video_length=500
    )

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=exp_dir / "models",
        log_path=exp_dir / "logs",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    # Create PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        normalize_advantage=config["normalize_advantage"],
        tensorboard_log=exp_dir / "logs",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            ),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
    )

    try:
        # Train the model
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=eval_callback,
            progress_bar=True
        )

        # Save final model
        model.save(exp_dir / "models" / "final_model")

    finally:
        # Clean up
        env.close()
        eval_env.close()

    # Create a new environment for final testing with human rendering
    test_env = gym.make("PandaSlideDense", render_mode="human")

    try:
        obs, _ = test_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = test_env.step(action)
            if terminated or truncated:
                obs, _ = test_env.reset()
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
