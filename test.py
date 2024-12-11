from hurry_taxi.envs.taxi_grid import TaxiGridEnv

env = TaxiGridEnv()
env._init_map()
env.show_grid_map()