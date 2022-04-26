from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt


sizes = [20, 100]
repetitions = 1000
names = ["Normal", "Cauchy", "Laplace", "Poisson", "Uniform"]


def mustache(sample):
    q_1, q_3 = np.quantile(sample, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)


def count_emissions(sample):
    x1, x2 = mustache(sample)
    filtered = [x for x in sample if x > x2 or x < x1]
    return len(filtered)


def draw_boxplot(parts, name):
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=parts, palette='Set1', orient='h')
    sns.despine(offset=10)
    plt.xlabel("x")
    plt.ylabel("n")
    plt.title(name + " distribution")
    plt.show()


def emission_share(distribution, size, name):
    count = 0
    for i in range(repetitions):
        sample = distribution.rvs(size=size)
        sample.sort()
        count += count_emissions(sample)
    count /= (size * repetitions)
    print("Доля выбросов для выборки для " + name + " распределения из " + str(size) + " элементов: " + str(count))


def build_boxplot(distribution, name):
    parts = []
    for size in sizes:
        emission_share(distribution, size, name)

        sample = distribution.rvs(size=size)
        sample.sort()
        parts.append(sample)
    draw_boxplot(parts, name)


build_boxplot(norm(), names[0])
build_boxplot(cauchy(), names[1])
build_boxplot(laplace(loc=0, scale=1 / math.sqrt(2)), names[2])
build_boxplot(poisson(10), names[3])
build_boxplot(uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)), names[4])