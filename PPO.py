import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import os

import hurry_taxi


def make_env():
    env = gym.make(
        "hurry_taxi/TaxiGrid-v0",
        max_steps=args.steps,
        agents_number=args.agents,
        npc_number=args.npcs,
        grid_size=args.size,
    )
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=1000, required=False)
    parser.add_argument(
        "--size", type=int, default=25, required=False, choices=[5, 10, 25]
    )
    parser.add_argument("--agents", type=int, default=4, required=False)
    parser.add_argument("--npcs", type=int, default=4, required=False)

    args = parser.parse_args()

    model_name = f"ppo_{args.size}_{args.steps}_{args.agents}_{args.npcs}"
    folder_name = os.path.join("logs", model_name)
    os.makedirs(folder_name, exist_ok=True)

    env = make_vec_env(
        make_env,
        n_envs=16,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=folder_name,
    )

    # TODO: Correr este modelo
    model = PPO(
        "MlpPolicy", env, verbose=1, gamma=0.99, learning_rate=0.0001, device="cpu"
    )
    model.learn(total_timesteps=10000000, log_interval=10)
    model.save(os.path.join("models", model_name))
