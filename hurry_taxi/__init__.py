from gymnasium.envs.registration import register

register(
    id="hurry_taxi/TaxiGrid-v0",
    entry_point="hurry_taxi.envs:TaxiGridEnv",
)