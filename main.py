import gymnasium
import hurry_taxi
from hurry_taxi.envs.taxi_grid import Actions

import pygame

def handle_player_input():
    global quit_game
    keys = pygame.key.get_pressed()
    action = [1, 1]
    if keys[pygame.K_RIGHT]:
        action[1] = -1
    if keys[pygame.K_UP]:
        action[1] = -0.5
    if keys[pygame.K_LEFT]:
        action[1] = 0
    if keys[pygame.K_DOWN]:
        action[1] = 0.5
    if keys[pygame.K_d]:
        action[0] = -1
    if keys[pygame.K_w]:
        action[0] = -0.5
    if keys[pygame.K_a]:
        action[0] = 0
    if keys[pygame.K_s]:
        action[0] = 0.5
    if keys[pygame.K_ESCAPE]:
        quit_game = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_game = True

    return action


quit_game = False
env = gymnasium.make("hurry_taxi/TaxiGrid-v0", render_mode="human")
observation, info = env.reset()
done = False
while not done and not quit_game:
    action = handle_player_input()
    observation, reward, done, _, info = env.step(action)
env.close()