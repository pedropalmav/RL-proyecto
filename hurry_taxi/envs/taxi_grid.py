import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np
from enum import Enum

from hurry_taxi.utils.guaussian import Gaussian2D
from hurry_taxi.utils.position_randomizer import PositionRandomizer
from hurry_taxi.envs.map import map


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

    def __init__(self, render_mode=None, max_steps=100):
        self.grid_size = 25
        self.window_size = 512
        self.max_steps = max_steps
        ##grid
        self.map = map
        # Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=self.grid_size, shape=(2, ), dtype=int),
            spaces.Discrete(4), # Direction of road: east, north, west, south
            spaces.Discrete(2), # Has passenger: yes, no
        ))

        self._action_to_vector = {
            Actions.right: np.array([1, 0]),
            Actions.up: np.array([0, -1]),
            Actions.left: np.array([-1, 0]),
            Actions.down: np.array([0, 1]),
            Actions.nothing: np.array([0, 0]),
        }

        self._direction_to_angle = {
            Directions.north: 0,
            Directions.east: -90,
            Directions.south: 180,
            Directions.west: 90,
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
        return (self._agent_location, self._direction.value, self._has_passenger)
    
    def _get_info(self):
        return {
            "target": self._target_location
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self._agent_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
        self._direction = Directions.east # TODO: manejar según la dirección de la calle
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

        self._direction = self._get_direction_from_action(action)
        terminated = self.steps >= self.max_steps or self._event == Events.collision

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_reward(), terminated, False, self._get_info()
    
    def _get_direction_from_action(self, action):
        match action:
            case Actions.right:
                return Directions.east
            case Actions.up:
                return Directions.north
            case Actions.left:
                return Directions.west
            case Actions.down:
                return Directions.south
            case _:
                return self._direction
    
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
        
        road_folder = "hurry_taxi/assets/roads/"
        self.road_sprite = {
            'horizontal': pygame.image.load(road_folder+'horizontal_road.png').convert_alpha(),
            'vertical': pygame.image.load(road_folder+'vertical_road.png').convert_alpha(),
            'curve_up_right': pygame.image.load(road_folder+'curve01.png').convert_alpha(),
            'curve_up_left': pygame.image.load(road_folder+'curve02.png').convert_alpha(),
            'curve_down_right': pygame.image.load(road_folder+'curve03.png').convert_alpha(),
            'curve_down_left': pygame.image.load(road_folder+'curve04.png').convert_alpha(),
            'crossroad': pygame.image.load(road_folder+'crossroad.png').convert_alpha(),
            'T_down': pygame.image.load(road_folder+'t_intersection03.png').convert_alpha(),
            'T_up': pygame.image.load(road_folder+'t_intersection02.png').convert_alpha(),
            'T_right': pygame.image.load(road_folder+'t_intersection04.png').convert_alpha(),
            'T_left': pygame.image.load(road_folder+'t_intersection01.png').convert_alpha(),
            'end_down': pygame.image.load(road_folder+'end_road01.png').convert_alpha(),
            'end_up': pygame.image.load(road_folder+'end_road02.png').convert_alpha(),
            'end_right': pygame.image.load(road_folder+'end_road03.png').convert_alpha(),
            'end_left': pygame.image.load(road_folder+'end_road04.png').convert_alpha(),
            'building': pygame.image.load('hurry_taxi/assets/grass.png').convert_alpha(),
        }
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        if not hasattr(self, "_car_sprite"):
            self._car_sprite = pygame.image.load("hurry_taxi/assets/cars/taxi_small.png").convert_alpha()

        if not hasattr(self, "_person_sprite"):
            self._person_sprite = pygame.image.load("hurry_taxi/assets/characters/character_black_blue.png").convert_alpha()
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                position = (int(x * pix_square_size), int(y * pix_square_size))
                if self.map[x][y] == 1:  # Road
                    connections = self.get_connections(x, y)
                    sprite = self.get_sprite(connections)
                    if sprite:
                        canvas.blit(sprite, position)
                else:  # Building
                    canvas.blit(self.road_sprite['building'], position)


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
        rotated_car_sprite = pygame.transform.rotate(self._car_sprite, self._direction_to_angle[self._direction])
        canvas.blit(rotated_car_sprite, agent_position)

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

    
    def get_sprite(self, connections):
        road_type = self.get_road_type(connections)
        return self.road_sprite.get(road_type)
    def get_road_type(self, connections):
        connection_count = sum(connections.values())
        if connection_count == 2:
            if connections["up"] and connections["right"]:
                return "curve_up_right"
            elif connections["up"] and connections["left"]:
                return "curve_up_left"
            elif connections["down"] and connections["right"]:
                return "curve_down_right"
            elif connections["down"] and connections["left"]:
                return "curve_down_left"
            elif connections["up"] and connections["down"]:
                return "vertical"
            elif connections["left"] and connections["right"]:
                return "horizontal"
        elif connection_count == 4:
            return "crossroad"  # Connected to all 4 sides
        elif connection_count == 3:
            if not connections["up"]:
                return "T_down"
            elif not connections["down"]:
                return "T_up"
            elif not connections["left"]:
                return "T_right"
            elif not connections["right"]:
                return "T_left"
        elif connection_count == 1:
            if connections["up"]:
                return "end_down"
            elif connections["down"]:
                return "end_up"
            elif connections["left"]:
                return "end_right"
            elif connections["right"]:
                return "end_left"
        return "building"

    def get_connections(self, x, y):
        neighbors = {
            "up": (x - 1, y),
            "down": (x + 1, y),
            "left": (x, y - 1),
            "right": (x, y + 1)
        }
        connections = {
            direction: self.map[nx][ny] if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size else 0
            for direction, (nx, ny) in neighbors.items()
        }
        return connections


            
