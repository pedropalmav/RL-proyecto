import os
import pandas as pd

class LogsLoader:
    @staticmethod
    def load_data(models, size, steps, agents, npcs):
        data = {}
        for model in models:
            path = os.path.join('logs', f'{model}_{size}_{steps}_{agents}_{npcs}.monitor.csv')
            model_data = pd.read_csv(path, skiprows=1, usecols=['l', 'r'])
            data[model] = model_data
        return data