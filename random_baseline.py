import hurry_taxi
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym


def get_random_action(agents_number=1):
    return np.random.uniform(-1, 1, agents_number)


steps = 5000
size = 5
agents = 1
npcs = 0

env = gym.make(
    "hurry_taxi/TaxiGrid-v0",
    max_steps=steps,
    agents_number=agents,
    npc_number=npcs,
    grid_size=size,
    max_episode_steps=steps,
)

results = []
for i in range(700000 // steps):
    obs, info = env.reset()
    cumulative_reward = 0
    done = False
    while not done:
        action = get_random_action(agents)
        obs, rewards, done, truncated, info = env.step(action)
        cumulative_reward += rewards
    results.append(cumulative_reward)

data = pd.DataFrame(results, columns=["r"])
plt.figure(figsize=(10, 5))
training_steps = data.index * steps
plt.plot(training_steps, data["r"])
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.show()
