import os
import pandas as pd


class LogsLoader:
    @staticmethod
    def load_data(models, size, steps, agents, npcs):
        data = {}
        for model in models:
            path = os.path.join(
                "logs", f"{model}_{size}_{steps}_{agents}_{npcs}.monitor.csv"
            )
            model_data = pd.read_csv(path, skiprows=1, usecols=["l", "r"])
            data[model] = model_data
        return data

    @staticmethod
    def load_vectorized_logs(model, size, episode_steps, agents, npcs):
        logs = []
        path = os.path.join("logs", f"{model}_{size}_{episode_steps}_{agents}_{npcs}")
        for file in os.listdir(path):
            if file.endswith(".monitor.csv"):
                logs.append(
                    pd.read_csv(
                        os.path.join(path, file), skiprows=1, usecols=["l", "r"]
                    )
                )
        return logs


if __name__ == "__main__":
    ppo_logs = LogsLoader.load_vectorized_logs("ppo", 5, 5000, 1, 0)
    for log in ppo_logs:
        print(log)
