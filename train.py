import gymnasium as gym
from stable_baselines3 import PPO
import hurry_taxi

env = gym.make("hurry_taxi/TaxiGrid-v0")
obs, info = env.reset()


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("test_PPO_1")

