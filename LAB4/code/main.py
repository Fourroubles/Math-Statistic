from scipy.stats import norm, laplace, poisson, cauchy, uniform
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import math as m
import seaborn as sns
import matplotlib.pyplot as plt


sizes = [20, 60, 100]
koeffs = [0.5, 1, 2]
left, right = -4, 4
poisson_left, poisson_right = 6, 14
names = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]
count = 5


def get_samples(size):
    return [norm.rvs(size=size), cauchy.rvs(size=size), laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0),
            poisson.rvs(10, size=size), uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))]


def get_densities(x):
    return [norm.pdf(x), cauchy.pdf(x), laplace.pdf(x, loc=0, scale=1 / m.sqrt(2)), poisson(10).pmf(x),
            uniform.pdf(x, loc=-m.sqrt(3), scale=2 * m.sqrt(3))]


def get_cdf(x):
    cdf_list = [norm.cdf(x), cauchy.cdf(x), laplace.cdf(x, loc=0, scale=1 / m.sqrt(2)), poisson.cdf(x, mu=10),
                uniform.cdf(x, loc=-m.sqrt(3), scale=2 * m.sqrt(3))]
    return cdf_list


def draw_graphics():
    sns.set_style('whitegrid')
    for number in range(count):
        figures, axs = plt.subplots(ncols=3, figsize=(15, 5))

        for size in range(len(sizes)):
            samples = get_samples(sizes[size])
            sample = samples[number]
            ecdf = ECDF(sample)

            if number != 3:
                x = np.linspace(left, right, 1000)
            else:
                x = np.linspace(poisson_left, poisson_right, 1000)
            y = get_cdf(x)

            axs[size].plot(x, y[number], color='red', label='cdf')
            axs[size].plot(x, ecdf(x), color='blue', label='ecdf')
            axs[size].set(xlabel='x', ylabel='F(x)')
            axs[size].set_title("n = " + str(sizes[size]))
        figures.suptitle(names[number] + " distribution")
        plt.show()
    return


def draw_kde():
    sns.set_style('whitegrid')
    for number in range(count):
        for size in range(len(sizes)):
            figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
            samples = get_samples(sizes[size])
            sample = samples[number]

            if number != 3:
                x = np.linspace(left, right, 1000)
                start, stop = left, right
            else:
                x = np.linspace(poisson_left, poisson_right, -poisson_left + poisson_right + 1)
                start, stop = poisson_left, poisson_right

            for koeff in range(len(koeffs)):
                y = get_densities(x)
                axs[koeff].plot(x, y[number], color="red", label="pdf")
                sns.kdeplot(data=sample, bw_method="silverman", bw_adjust=koeffs[koeff], ax=axs[koeff],
                            fill=True, linewidth=0, label="kde")
                axs[koeff].set(xlabel="x", ylabel="f(x)")
                axs[koeff].set_xlim([start, stop])
                axs[koeff].set_title("h = " + str(koeffs[koeff]))
            figures.suptitle(names[number] + " n = " + str(sizes[size]))
            plt.show()


draw_graphics()
draw_kde()