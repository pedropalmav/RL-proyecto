import hurry_taxi
import os
import gymnasium as gym
from stable_baselines3 import PPO

steps = 5000
size = 10
agents = 2
npcs = 2

model = PPO.load(os.path.join("models", f"ppo_{size}_{steps}_{agents}_{npcs}"))

env = gym.make("hurry_taxi/TaxiGrid-v0", max_steps=steps, agents_number=agents, npc_number=npcs, grid_size=size, render_mode="human")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        break