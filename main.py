import gymnasium
import hurry_taxi

import pygame

def handle_player_input():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        return 0
    if keys[pygame.K_UP]:
        return 1
    if keys[pygame.K_LEFT]:
        return 2
    if keys[pygame.K_DOWN]:
        return 3
    return 4        


env = gymnasium.make("hurry_taxi/TaxiGrid-v0", render_mode="human")
observation, info = env.reset()
done = False
while not done:
    action = handle_player_input()
    observation, reward, done, _, info = env.step(action)
    # print(observation, reward)
env.close()