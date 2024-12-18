import os
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    models = ["a2c", "ppo", "sac", "td3"]

    @staticmethod
    def plot_line(data, size, steps, agents, npcs):
        plt.figure(figsize=(10, 5))
        for model in data.keys():
            training_steps = data[model].index * steps
            plt.plot(training_steps, data[model]["r"], label=model)
        plt.xlabel("Training Steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(os.path.join("imgs", f"reward_{size}_{steps}_{agents}_{npcs}.png"))

    @staticmethod
    def line_plot_with_bands(data, size, steps, agents, npcs):
        grouped = data.groupby("episode")
        mean_rewards = grouped["r"].mean()
        std_rewards = grouped["r"].std()

        plt.figure(figsize=(10, 6))
        plt.plot(mean_rewards)
        # TODO: fix this
        plt.fill_between(
            mean_rewards.index,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        plt.show()


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_line(1000, 2, 4)
