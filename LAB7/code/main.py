import numpy as np
from scipy.stats import laplace, uniform
from tabulate import tabulate
import scipy.stats as stats
import math


alpha = 0.05
p = 1 - alpha


def probability(sample, limits):
    p_list = np.array([])
    n_list = np.array([])

    for i in range(-1, len(limits)):
        if i == -1:
            previous_cdf = 0
        else:
            previous_cdf = stats.norm.cdf(limits[i])
        if i == len(limits) - 1:
            current_cdf = 1
        else:
            current_cdf = stats.norm.cdf(limits[i + 1])
        p_list = np.append(p_list, current_cdf - previous_cdf)

        if i == -1:
            n_list = np.append(n_list, len(sample[sample <= limits[0]]))
        elif i == len(limits) - 1:
            n_list = np.append(n_list, len(sample[sample >= limits[-1]]))
        else:
            n_list = np.append(n_list, len(sample[(sample <= limits[i + 1]) & (sample >= limits[i])]))

    return n_list, p_list


def latex_table(n_list, p_list, size, limits):
    result = np.divide(np.multiply((n_list - size * p_list), (n_list - size * p_list)), p_list * size)
    rows = []

    for i in range(0, len(n_list)):
        if i == 0:
            boarders = ["-inf", np.around(limits[0], decimals=4)]
        elif i == len(n_list) - 1:
            boarders = [np.around(limits[-1], decimals=4), "inf"]
        else:
            boarders = [np.around(limits[i - 1], decimals=4), np.around(limits[i], decimals=4)]

        rows.append([i + 1, boarders, n_list[i], np.around(p_list[i], decimals=4),
                     np.around(p_list[i] * size, decimals=4), np.around(n_list[i] - size * p_list[i], decimals=4),
                     np.around(result[i], decimals=4)])

    rows.append([len(n_list) + 1, "-", np.sum(n_list), np.around(np.sum(p_list), decimals=4),
                 np.around(np.sum(p_list * size), decimals=4), np.around(np.sum(n_list - size * p_list), decimals=4),
                 np.around(np.sum(result), decimals=4)])

    print(tabulate(rows, tablefmt="latex"))


def run(sample, size):
    mu = np.mean(sample)
    sigma = np.std(sample)
    k = math.ceil(1.72 * size ** (1 / 3))
    chi_2 = stats.chi2.ppf(p, k - 1)
    print('mu = ' + str(np.around(mu, decimals=4)))
    print('sigma = ' + str(np.around(sigma, decimals=4)))
    print('chi_2 = ' + str(chi_2))

    limits = np.linspace(-1.1, 1.1, num=k - 1)
    n_list, p_list = probability(sample, limits)
    latex_table(n_list, p_list, size, limits)


run(np.random.normal(0, 1, size=100), 100)
run(laplace.rvs(size=20, scale=1 / math.sqrt(2), loc=0), 20)