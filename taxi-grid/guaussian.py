import numpy as np

class Gaussian2D:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def get_sample(self):
        return np.random.multivariate_normal(self.mean, self.cov)
    
    def get_multiple_samples(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, n)
    
if __name__ == "__main__":
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    g = Gaussian2D(mean, cov)

    data = g.get_multiple_samples(1000)
    x, y = data[:, 0], data[:, 1]

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.5, edgecolors='k')
    plt.grid(True)
    plt.show()