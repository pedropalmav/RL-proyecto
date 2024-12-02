import random

class PositionRandomizer:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def discrete_randomize(self):
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
    
    def continuous_randomize(self):
        return (random.random() * self.grid_size, random.random() * self.grid_size)