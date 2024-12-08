import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np
from enum import Enum

from hurry_taxi.utils.guaussian import Gaussian2D
from hurry_taxi.utils.position_randomizer import PositionRandomizer

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Directions(Enum):
    east = 0
    north = 1
    west = 2
    south = 3

class TaxiGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=5, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=grid_size, shape=(2, ), dtype=int),
            spaces.Discrete(4) # Direction of road: east, north, west, south
        ))
        self.state = None

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        self._init_randomizers()
        self._init_visualization()
    
    def _init_randomizers(self):
        self.randomizer = PositionRandomizer(self.grid_size)
        mean = self.grid_size / 2
        self.gaussian = Gaussian2D([mean, mean], [[1, 0], [0, 1]])

    def _init_visualization(self):
        self.render_mode = "human"
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.state = (np.array(self.randomizer.discrete_randomize(), dtype=int), 0)
        self._target_location = self.randomizer.discrete_randomize()

        return self.state, {}
    
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