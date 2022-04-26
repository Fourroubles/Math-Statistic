from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math


def build_distribution():
    names = ["Normal distribution", "Cauchy distribution", "Laplace distribution", "Poisson distribution", "Uniform distribution"]
  
    build_histogram(norminvgauss(1, 0), "deepskyblue", names[0])
    build_histogram(cauchy(), "deepskyblue", names[1])
    build_histogram(laplace(loc=0, scale=1 / math.sqrt(2)), "deepskyblue", names[2])
    build_histogram(poisson(10), "deepskyblue", names[3])
    build_histogram(uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)), "deepskyblue", names[4])


def build_histogram(distribution, color, name):
    sizes = [10, 50, 1000]
    labels = ["size", "distribution"]
    line_type = "k--"

    for size in sizes:
        numbers = distribution.rvs(size)
        fig, ax = plt.subplots(1, 1)
        ax.hist(numbers, density=True, alpha=0.7, color=color)
        if name == "Poisson distribution":
            x = np.arange(poisson.ppf(0.001, 10), poisson.ppf(0.999, 10))
            ax.plot(x, distribution.pmf(x), line_type)
        else:
            x = np.linspace(distribution.ppf(0.001), distribution.ppf(0.999), 100)
            ax.plot(x, distribution.pdf(x), line_type)
        ax.set_xlabel(labels[0] + ": " + str(size))
        ax.set_ylabel(labels[1])
        ax.set_title(name)
        plt.grid()
        plt.show()


build_distribution()