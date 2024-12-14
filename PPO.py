import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import argparse
import os

import hurry_taxi

parser = argparse.ArgumentParser()

parser.add_argument("--steps", type=int, default=1000, required=False)
parser.add_argument("--size", type=int, default=25, required=False, choices=[5, 10, 25])
parser.add_argument("--agents", type=int, default=4, required=False)
parser.add_argument("--npcs", type=int, default=4, required=False)

args = parser.parse_args()

filename = f"ppo_{args.size}_{args.steps}_{args.agents}_{args.npcs}"

env = gym.make(
    "hurry_taxi/TaxiGrid-v0", 
    max_steps=args.steps, 
    agents_number=args.agents, 
    npc_number=args.npcs,
    grid_size=args.size
)

env = Monitor(env, filename=os.path.join("logs", filename))
obs, info = env.reset()


model = PPO("MlpPolicy", env, verbose=1, gamma=0.9)
model.learn(total_timesteps=25*args.steps, log_interval=10)
model.save(os.path.join("models", filename))
