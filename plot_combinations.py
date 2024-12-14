from plotter import Plotter
from logs_loader import LogsLoader

models = ['ppo']
size = 10
steps = [5000]
agents = [2]
npcs = [2]

for step in steps:
    for agent in agents:
        for npc in npcs:
            data = LogsLoader.load_data(models, size, step, agent, npc)
            Plotter.plot_line(data, size, step, agent, npc)