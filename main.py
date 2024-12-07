import gymnasium
import hurry_taxi

env = gymnasium.make("hurry_taxi/TaxiGrid-v0")
print(env.reset())