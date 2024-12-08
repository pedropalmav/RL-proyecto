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

class Events(Enum):
    takes_passenger = 0
    leaves_passenger = 1
    collision = 2

class TaxiGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=5, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=grid_size, shape=(2, ), dtype=int),
            spaces.Discrete(4), # Direction of road: east, north, west, south
            spaces.Discrete(2), # Has passenger: yes, no
        ))

        self._action_to_vector = {
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

    def _get_obs(self):
        return (self._agent_location, self._direction, self._has_passenger)
    
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self._agent_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
        self._direction = 0 # TODO: manejar según la dirección de la calle
        self._has_passenger = 0
        self._target_location = self.randomizer.discrete_randomize()

        return self._get_obs(), {}
    
    def step(self, action):
        self._event = None
        action_vector = self._action_to_vector[action]
        new_location = self._agent_location + action_vector
        
        self._handle_collision(new_location)
        self._handle_passenger()

        self._direction = action
        terminated = self.steps >= self.max_steps

        return self._get_obs(), self._get_reward(), terminated, False, self._get_info()
    
    def _handle_collision(self, new_location):
        if not self._is_off_limits(new_location):
            self._agent_location = new_location
        # TODO: Colision con limite de calle
        # TODO: Colision con otro auto
        # TODO: Reward = -2

    def _is_off_limits(self, location):
        return location[0] < 0 or location[0] >= self.grid_size \
            or location[1] < 0 or location[1] >= self.grid_size
    
    def _handle_passenger(self):
        if self._is_target_near() and not self._has_passenger:
            self._has_passenger = 1
            self._target_location = self.randomizer.discrete_randomize()
            self._event = Events.takes_passenger
        elif self._is_target_near() and self._has_passenger:
            self._has_passenger = 0
            self._target_location = self.randomizer.discrete_randomize()
            self._event = Events.leaves_passenger

    def _is_target_near(self):
        return np.linalg.norm(self._agent_location - self._target_location, ord=1) == 1


    def _get_reward(self):
        match(self._event):
            case Events.takes_passenger:
                return 1
            case Events.leaves_passenger:
                return 2
            case Events.collision:
                return -2
            case _:
                return 0

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