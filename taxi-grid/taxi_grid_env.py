import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from guaussian import Gaussian2D
from position_randomizer import PositionRandomizer

class TaxiGridEnv(gym.Env):
    def __init__(self, grid_size=4, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        # TODO: Revisar este espacio de observación
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32)
        self.state = None

        self.randomizer = PositionRandomizer(self.grid_size)
        mean = grid_size / 2
        self.gaussian = Gaussian2D([mean, mean], [[1, 0], [0, 1]])

    def reset(self):
        self.steps = 0
        self.state = PositionRandomizer(self.grid_size).continuous_randomize()
        return self.state, {}
    
    def step(self, action):
        pass

    def render(self):
        pass