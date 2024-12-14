import os
import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    models = ['a2c', 'ppo', 'sac', 'td3']

    @staticmethod
    def plot_line(data, size, steps, agents, npcs):
        plt.figure(figsize=(10, 5))
        for model in data.keys():
            training_steps = data[model].index * steps
            plt.plot(training_steps, data[model]['r'], label=model)
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join('imgs', f'reward_{size}_{steps}_{agents}_{npcs}.png'))

if __name__ == '__main__':
    plotter = Plotter()
    plotter.plot_line(1000, 2, 4)