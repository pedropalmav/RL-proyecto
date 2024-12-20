import hurry_taxi
from hurry_taxi.envs.taxi_grid import Actions
import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import PPO


def parse_action(actions):
    discrete_actions = []
    for action in actions:
        discrete_action = int(
            np.clip((action + 1) * (len(Actions) - 1) / 2, 0, len(Actions) - 1)
        )
        discrete_actions.append(discrete_action)
    return [Actions(action) for action in discrete_actions]


steps = 5000
size = 5
agents = 1
npcs = 0

model = PPO.load(os.path.join("models", f"ppo_{size}_{steps}_{agents}_{npcs}"))

env = gym.make(
    "hurry_taxi/TaxiGrid-v0",
    max_steps=steps,
    agents_number=agents,
    npc_number=npcs,
    grid_size=size,
    render_mode="human",
    max_episode_steps=steps,
)

obs, info = env.reset()
i = 0
cumulative_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    cumulative_reward += rewards
    i += 1
    env.render()
    print(i, parse_action(action), cumulative_reward)
    if terminated:
        break

env.close()
