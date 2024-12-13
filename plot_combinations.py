from plotter import Plotter

steps = [1000]
agents = [2, 4, 6]
npcs = [4, 8]

for step in steps:
    for agent in agents:
        for npc in npcs:
            Plotter.plot_line(step, agent, npc)