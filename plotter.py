import os
import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    models = ['a2c', 'ppo', 'sac', 'td3']

    @staticmethod
    def plot_line(steps, agents, npcs):
        data = Plotter.load_data(steps, agents, npcs)
        plt.figure(figsize=(10, 5))
        for model in Plotter.models:
            training_steps = data[model].index * steps
            plt.plot(training_steps, data[model]['r'], label=model)
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join('imgs', f'reward_{steps}_{agents}_{npcs}.png'))

    @staticmethod
    def load_data(steps, agents, npcs):
        data = {}
        for model in Plotter.models:
            path = os.path.join('logs', f'{model}_{steps}_{agents}_{npcs}.monitor.csv')
            model_data = pd.read_csv(path, skiprows=1, usecols=['l', 'r'])
            data[model] = model_data
        return data
    
if __name__ == '__main__':
    plotter = Plotter()
    plotter.plot_line(1000, 2, 4)