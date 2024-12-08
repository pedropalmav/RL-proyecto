import gymnasium
import hurry_taxi

import pygame

def handle_player_input():
    from hurry_taxi.envs.taxi_grid import Actions
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        return Actions.right
    if keys[pygame.K_UP]:
        return Actions.up
    if keys[pygame.K_LEFT]:
        return Actions.left
    if keys[pygame.K_DOWN]:
        return Actions.down
    return Actions.nothing


env = gymnasium.make("hurry_taxi/TaxiGrid-v0", render_mode="human")
observation, info = env.reset()
done = False
while not done:
    action = handle_player_input()
    observation, reward, done, _, info = env.step(action)
    # print(observation, reward)
env.close()