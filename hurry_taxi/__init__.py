from gymnasium.envs.registration import register

register(
    id="hurry_taxi/GridWorld-v0",
    entry_point="hurry_taxi.envs:GridWorldEnv",
)
