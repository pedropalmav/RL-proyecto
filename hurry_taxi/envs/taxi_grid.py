import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np

from hurry_taxi.utils.guaussian import Gaussian2D
from hurry_taxi.utils.position_randomizer import PositionRandomizer

class TaxiGridEnv(gym.Env):
    def __init__(self, grid_size=4, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        # TODO: Revisar este espacio de observación
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(2, ), dtype=np.float32)
        self.state = None

        self.randomizer = PositionRandomizer(self.grid_size)
        mean = grid_size / 2
        self.gaussian = Gaussian2D([mean, mean], [[1, 0], [0, 1]])

        self.render_mode = "human"
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.state = PositionRandomizer(self.grid_size).continuous_randomize()
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        pass

    def render(self):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed. Use `pip install pygame` to install it."
            ) from e
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()
    
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.isopen = False