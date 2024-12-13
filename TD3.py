import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
import hurry_taxi

env = gym.make("hurry_taxi/TaxiGrid-v0", max_steps = 1000, agents_number=4)

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

env = Monitor(env, filename='logs/td3')
obs, info = env.reset()


model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000, log_interval=10)
model.save("test_PPO_1")

