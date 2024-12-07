from gymnasium.envs.registration import register

register(
    id="TaxiGriding/GridWorld-v0",
    entry_point="TaxiGriding.envs:GridWorldEnv",
)
