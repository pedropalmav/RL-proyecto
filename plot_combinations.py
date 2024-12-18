from plotter import Plotter
from logs_loader import LogsLoader
import pandas as pd

models = ["ppo"]
size = 5
steps = [5000]
agents = [1]
npcs = [0]

for step in steps:
    for agent in agents:
        for npc in npcs:
            logs = LogsLoader.load_vectorized_logs("ppo", size, step, agent, npc)
            data = pd.concat(logs, keys=range(len(logs)), names=["agent", "episode"])
            data.reset_index(inplace=True)
            Plotter.line_plot_with_bands(data, size, step, agent, npc)
