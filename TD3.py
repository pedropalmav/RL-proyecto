import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
import argparse
import os
import hurry_taxi

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000, required=False)
parser.add_argument("--agents", type=int, default=4, required=False)
parser.add_argument("--npcs", type=int, default=4, required=False)
args = parser.parse_args()

filename = f"td3_{args.steps}_{args.agents}_{args.npcs}"

env = gym.make("hurry_taxi/TaxiGrid-v0", max_steps=args.steps, agents_number=args.agents, npc_number=args.npcs)

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

env = Monitor(env, filename=os.path.join("logs", filename))
obs, info = env.reset()


model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25*args.steps, log_interval=10)
model.save(os.path.join("models", filename))

