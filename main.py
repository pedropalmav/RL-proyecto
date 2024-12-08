import gymnasium
import hurry_taxi

env = gymnasium.make("hurry_taxi/TaxiGrid-v0", render_mode="human")
# env = gymnasium.make("hurry_taxi/GridWorld-v0", render_mode="human")
observation, info = env.reset()
print(observation)
env.close()