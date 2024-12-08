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
    nothing = 4

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
        self.window_size = 512
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
            Actions.up.value: np.array([0, -1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, 1]),
            Actions.nothing.value: np.array([0, 0]),
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
        self.window = None
        self.screen = None
        self.clock = None
        self.isopen = True

    def _get_obs(self):
        return (self._agent_location, self._direction, self._has_passenger)
    
    def _get_info(self):
        return {
            "target": self._target_location
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self._agent_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
        self._direction = 0 # TODO: manejar según la dirección de la calle
        self._has_passenger = 0
        self._target_location = np.array(self.randomizer.discrete_randomize(), dtype=int)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}
    
    def step(self, action):
        self._event = None
        action_vector = self._action_to_vector[action]
        new_location = self._agent_location + action_vector
        
        self._handle_collision(new_location)
        self._handle_passenger()

        self._direction = action
        terminated = self.steps >= self.max_steps or self._event == Events.collision

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_reward(), terminated, False, self._get_info()
    
    def _handle_collision(self, new_location):
        if self._agent_collides(new_location):
            self._event = Events.collision
        else:
            self._agent_location = new_location

    def _agent_collides(self, location):
        # TODO: colisión con bordes de la calle
        # TODO: colisión con otro auto
        return self._is_off_limits(location)

    def _is_off_limits(self, location):
        return location[0] < 0 or location[0] >= self.grid_size \
            or location[1] < 0 or location[1] >= self.grid_size
    
    def _handle_passenger(self):
        if self._is_target_near() and not self._has_passenger:
            self._has_passenger = 1
            self._target_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
            self._event = Events.takes_passenger
        elif self._is_target_near() and self._has_passenger:
            self._has_passenger = 0
            self._target_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
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
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        if not hasattr(self, "_car_sprite"):
            self._car_sprite = pygame.image.load("hurry_taxi/assets/cars/taxi_small.png").convert_alpha()
            self._car_sprite = pygame.transform.scale(
                self._car_sprite, (int(pix_square_size), int(pix_square_size))
            )

        if not hasattr(self, "_person_sprite"):
            self._person_sprite = pygame.image.load("hurry_taxi/assets/characters/character_black_blue.png").convert_alpha()
            self._person_sprite = pygame.transform.scale(
                self._person_sprite, (int(pix_square_size), int(pix_square_size))
            )

        # Dibujar sprites
        target_position = (
            int(self._target_location[0] * pix_square_size),
            int(self._target_location[1] * pix_square_size),
        )
        canvas.blit(self._person_sprite, target_position)

        agent_position = (
            int(self._agent_location[0] * pix_square_size),
            int(self._agent_location[1] * pix_square_size),
        )
        canvas.blit(self._car_sprite, agent_position)

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.isopen = False