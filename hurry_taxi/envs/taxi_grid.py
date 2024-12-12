import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import os
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
        self.window_size = 1024
        self.max_steps = max_steps
        # TODO: refactorizar para utilizar numpy
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

        self.number_of_npcs = 4
        self.npcs = None

        self._init_randomizers()
        self._init_visualization(render_mode)
    
    def _init_randomizers(self):
        self.randomizer = PositionRandomizer(self.grid_size)
        mean = self.grid_size / 2
        self.gaussian = Gaussian2D([mean, mean], [[1, 0], [0, 1]])

    def _init_visualization(self, render_mode):
        self.render_mode = render_mode
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
        self._agent_location = self._get_location_on_road()
        self._direction = self._get_valid_direction(self._agent_location)
        self._has_passenger = 0
        self._target_location = self._get_valid_target_location()

        self._generate_npcs()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}
    
    def _get_location_on_road(self):
        while True:
            location = np.array(self.randomizer.discrete_randomize(), dtype=int)
            if not self._is_out_of_road(location):
                return location
            
    def _get_valid_direction(self, location):
        connections = self.get_connections(location[0], location[1])
        available_directions = []
        for direction, connected in connections.items():
            if connected:
                direction = self._get_direction_from_action(Actions[direction])
                available_directions.append(direction)
        return np.random.choice(available_directions)
    
    def _get_valid_target_location(self):
        while True:
            location = np.array(self.randomizer.discrete_randomize(), dtype=int)
            if self._is_beside_road(location) and not np.array_equal(location, self._agent_location):
                return location
            
    def _is_beside_road(self, location):
        connections = self.get_connections(location[0], location[1])
        return any(connections.values()) and self._is_out_of_road(location)
    
    def _generate_npcs(self):
        self.npcs = []
        for _ in range(self.number_of_npcs):
            npc_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
            while self._is_out_of_road(npc_location):
                npc_location = np.array(self.randomizer.discrete_randomize(), dtype=int)
            npc_color = np.random.choice(["black", "red", "blue", "green"])

            self.npcs.append({
                "location": npc_location,
                "direction": self._get_valid_direction(npc_location),
                "color": npc_color
            })
    
    def step(self, action):
        self._event = None
        action_vector = self._action_to_vector[action]
        new_location = self._agent_location + action_vector
        
        self._handle_collision(new_location)
        self._handle_passenger()

        self._move_npcs()

        self._direction = self._get_direction_from_action(action)
        terminated = self.steps >= self.max_steps or self._event == Events.collision

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_reward(), terminated, False, self._get_info()
    
    def _move_npcs(self):
        for npc in self.npcs:
            npc_location = npc["location"]
            npc_direction = npc["direction"]
            npc_action = self._get_npc_action(npc_location, npc_direction)
            npc_action_vector = self._action_to_vector[npc_action]
            npc["location"] = npc_location + npc_action_vector
            npc["direction"] = self._get_direction_from_action(npc_action)

    def _get_npc_action(self, location, direction):
        connections = self.get_connections(location[0], location[1])
        possible_actions = self._filter_actions_by_connections(connections)
        possible_actions = self._prevent_u_turn(possible_actions, direction)
        return np.random.choice(possible_actions)

    def _filter_actions_by_connections(self, connections):
        possible_actions = []
        for action in Actions:
            if action == Actions.nothing or connections[action.name] == 1:
                possible_actions.append(action)
        return possible_actions
    
    def _prevent_u_turn(self, actions, car_direction):
        opposite_direction = {
            Directions.east: Directions.west,
            Directions.west: Directions.east,
            Directions.north: Directions.south,
            Directions.south: Directions.north
        }
        if opposite_direction[car_direction] in actions:
            actions.remove(opposite_direction[car_direction])
        return actions
    
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
        return self._is_off_limits(location) or self._is_out_of_road(location) or self._hits_other_car(location)
    
    def _hits_other_car(self, location):
        for npc in self.npcs:
            if np.array_equal(npc["location"], location) and npc["direction"] == self._direction:
                return True
        return False
    
    def _is_out_of_road(self, location):
        return self.map[location[1]][location[0]] == 0

    def _is_off_limits(self, location):
        return location[0] < 0 or location[0] >= self.grid_size \
            or location[1] < 0 or location[1] >= self.grid_size
    
    def _handle_passenger(self):
        if self._is_target_near() and not self._has_passenger:
            self._has_passenger = 1
            self._target_location = self._get_valid_target_location()
            self._event = Events.takes_passenger
        elif self._is_target_near() and self._has_passenger:
            self._has_passenger = 0
            self._target_location = self._get_valid_target_location()
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
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        self._load_assets()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = (
            self.window_size / self.grid_size
        )

        self._render_background()
        self._render_roads()

        if not hasattr(self, "_person_sprite"):
            self._person_sprite = pygame.image.load("hurry_taxi/assets/characters/character_black_blue.png").convert_alpha()

        self._render_passenger()

        # TODO: meter dentro de render car
        agent_position = (
            int(self._agent_location[0] * self.pix_square_size),
            int(self._agent_location[1] * self.pix_square_size),
        )
        self._render_car(self.car_sprites["taxi"], agent_position, self._direction)
        
        self._render_npcs()


        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def _render_passenger(self):
        tile_position = (
            int(self._target_location[0] * self.pix_square_size),
            int(self._target_location[1] * self.pix_square_size),
        )
        connections = [direction for direction, connected in self.get_connections(*self._target_location).items() if connected]
        direction = np.random.choice(connections)
        passenger_position = self._get_passenger_position(tile_position, direction)
        person_sprite = pygame.transform.rotate(self._person_sprite, self._get_passenger_angle(direction))
        person_sprite = pygame.transform.scale(person_sprite, (int(self.pix_square_size) / 2, int(self.pix_square_size) / 2))
        self.canvas.blit(person_sprite, passenger_position)

    def _get_passenger_position(self, tile_position, direction):
        x, y = tile_position
        delta_to_middle = int(self.pix_square_size / 4)
        delta_to_side = int(self.pix_square_size / 2)
        match direction:
            case "right":
                return (x + delta_to_side, y + delta_to_middle)
            case "up":
                return (x + delta_to_middle, y)
            case "left":
                return (x, y + delta_to_middle)
            case "down":
                return (x + delta_to_middle, y + delta_to_side)
            case _:
                return tile_position
            
    def _get_passenger_angle(self, direction):
        match direction:
            case "right":
                return 90
            case "up":
                return 180
            case "left":
                return -90
            case "down":
                return 0
            case _:
                raise ValueError("Invalid direction")

    def _render_npcs(self):
        for npc in self.npcs:
            npc_position = (
                int(npc["location"][0] * self.pix_square_size),
                int(npc["location"][1] * self.pix_square_size),
            )
            self._render_car(self.car_sprites[npc["color"]], npc_position, npc["direction"])

    def _render_roads(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                position = (int(x * self.pix_square_size), int(y * self.pix_square_size))
                if self.map[y][x] == 1:
                    connections = self.get_connections(x, y)
                    sprite = self.get_sprite(connections)
                    sprite = pygame.transform.scale(sprite, (int(self.pix_square_size), int(self.pix_square_size)))
                    if sprite:
                        self.canvas.blit(sprite, position)

    def _render_background(self):
        background_sprite = pygame.transform.scale(self.road_sprite['grass'], (int(self.pix_square_size), int(self.pix_square_size)))
        sprite_width, sprite_height = background_sprite.get_size()
        canvas_width, canvas_height = self.canvas.get_size()

        for x in range(0, canvas_width, sprite_width):
            for y in range(0, canvas_height, sprite_height):
                self.canvas.blit(background_sprite, (x, y))
        
    def _load_assets(self):
        assets_folder = os.path.join("hurry_taxi", "assets")
        road_folder = os.path.join(assets_folder, "roads")
        cars_folder = os.path.join(assets_folder, "cars")
        self.road_sprite = {
            'horizontal': pygame.image.load(os.path.join(road_folder, 'horizontal_road.png')).convert_alpha(),
            'vertical': pygame.image.load(os.path.join(road_folder, 'vertical_road.png')).convert_alpha(),
            'curve_up_right': pygame.image.load(os.path.join(road_folder, 'curve03.png')).convert_alpha(),
            'curve_up_left': pygame.image.load(os.path.join(road_folder, 'curve04.png')).convert_alpha(),
            'curve_down_right': pygame.image.load(os.path.join(road_folder, 'curve01.png')).convert_alpha(),
            'curve_down_left': pygame.image.load(os.path.join(road_folder, 'curve02.png')).convert_alpha(),
            'crossroad': pygame.image.load(os.path.join(road_folder, 'crossroad.png')).convert_alpha(),
            'T_down': pygame.image.load(os.path.join(road_folder, 't_intersection02.png')).convert_alpha(),
            'T_up': pygame.image.load(os.path.join(road_folder, 't_intersection03.png')).convert_alpha(),
            'T_right': pygame.image.load(os.path.join(road_folder, 't_intersection01.png')).convert_alpha(),
            'T_left': pygame.image.load(os.path.join(road_folder, 't_intersection04.png')).convert_alpha(),
            'end_down': pygame.image.load(os.path.join(road_folder, 'end_road01.png')).convert_alpha(),
            'end_up': pygame.image.load(os.path.join(road_folder, 'end_road02.png')).convert_alpha(),
            'end_right': pygame.image.load(os.path.join(road_folder, 'end_road03.png')).convert_alpha(),
            'end_left': pygame.image.load(os.path.join(road_folder, 'end_road04.png')).convert_alpha(),
            'grass': pygame.image.load(os.path.join(assets_folder, 'grass.png')).convert_alpha(),
        }
        self.car_sprites = {
            'taxi': pygame.image.load(os.path.join(cars_folder, 'taxi_small.png')).convert_alpha(),
            'black': pygame.image.load(os.path.join(cars_folder, 'car_black_small.png')).convert_alpha(),
            'red': pygame.image.load(os.path.join(cars_folder, 'car_red_small.png')).convert_alpha(),
            'blue': pygame.image.load(os.path.join(cars_folder, 'car_blue_small.png')).convert_alpha(),
            'green': pygame.image.load(os.path.join(cars_folder, 'car_green_small.png')).convert_alpha(),
        }

    def _render_car(self, original_sprite, tile_position, direction):
        dimensions = self._get_car_dimensions(direction)
        position = self._get_car_position(tile_position, direction)
        sprite = pygame.transform.rotate(original_sprite, self._direction_to_angle[direction])
        sprite = pygame.transform.scale(sprite, dimensions)
        self.canvas.blit(sprite, position)
    
    def _get_car_dimensions(self, direction):
        if direction == Directions.east or direction == Directions.west:
            return (int(self.pix_square_size), int(self.pix_square_size) / 2)
        return (int(self.pix_square_size) / 2, int(self.pix_square_size))
    
    def _get_car_position(self, location, direction):
        x, y = location
        match direction:
            case Directions.east:
                return (x, y + int(self.pix_square_size / 2))
            case Directions.north:
                return (x + int(self.pix_square_size / 2), y)
            case _:
                return location
    
    def close(self):
        if self.screen is not None:
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
            if connections["left"] and connections["right"]:
                return "horizontal"
            elif connections["up"] and connections["down"]:
                return "vertical"
            elif connections["up"] and connections["right"]:
                return "curve_up_right"
            elif connections["up"] and connections["left"]:
                return "curve_up_left"
            elif connections["down"] and connections["right"]:
                return "curve_down_right"
            elif connections["down"] and connections["left"]:
                return "curve_down_left"
        elif connection_count == 4:
            return "crossroad" 
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
            "up": (x, y - 1),
            "down": (x, y + 1),
            "left": (x - 1, y),
            "right": (x + 1, y)
        }
        connections = {
            direction: self.map[ny][nx] if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size else 0
            for direction, (nx, ny) in neighbors.items()
        }
        return connections
