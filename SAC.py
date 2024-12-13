import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import hurry_taxi

env = gym.make("hurry_taxi/TaxiGrid-v0", max_steps = 1000, agents_number=4)
env = Monitor(env, filename='logs/sac')
obs, info = env.reset()


model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("test_PPO_1")

