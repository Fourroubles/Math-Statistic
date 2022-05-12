import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statistics
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse


sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
repetitions = 1000


def normal(size, rho):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size=size)


def mix_normal(size, rho):
    return 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) +\
           0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)


def quadrant_coeff(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n = [0, 0, 0, 0]

    for i in range(size):
        if x[i] >= med_x and y[i] >= med_y:
            n[0] += 1
        elif x[i] < med_x and y[i] >= med_y:
            n[1] += 1
        elif x[i] < med_x and y[i] < med_y:
            n[2] += 1
        elif x[i] >= med_x and y[i] < med_y:
            n[3] += 1
    return (n[0] + n[2] - n[1] - n[3]) / size


def coeffs(distribution, size, rho):
    p, s, q = [], [], []
    for i in range(repetitions):
        sample = distribution(size, rho)
        x, y = sample[:, 0], sample[:, 1]
        p.append(stats.pearsonr(x, y)[0])
        s.append(stats.spearmanr(x, y)[0])
        q.append(quadrant_coeff(x, y))
    return p, s, q


def results(pearson, spearman, quadrant):
    e, e_2, d = [], [], []
    e.append([np.around(np.median(pearson), decimals=4), np.around(np.median(spearman), decimals=4),
             np.around(np.median(quadrant), decimals=4)])
    p = np.median([pearson[k] ** 2 for k in range(repetitions)])
    s = np.median([spearman[k] ** 2 for k in range(repetitions)])
    q = np.median([quadrant[k] ** 2 for k in range(repetitions)])
    e_2.append([np.around(p, decimals=4), np.around(s, decimals=4), np.around(q, decimals=4)])
    d.append([np.around(statistics.variance(pearson), decimals=4), np.around(statistics.variance(spearman), decimals=4),
             np.around(statistics.variance(quadrant), decimals=4)])

    print(e, e_2, d)


def build_ellipse(x, y, ax):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', edgecolor='red')

    scale_x = np.sqrt(cov[0, 0]) * 3
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * 3
    mean_y = np.mean(y)

    transform = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)


def show_ellipse(size):
    fig, ax = plt.subplots(1, 3)
    titles = ["rho = 0", "rho = 0.5", "rho = 0.9"]

    for i in range(len(rhos)):
        sample = normal(size, rhos[i])
        x, y = sample[:, 0], sample[:, 1]
        build_ellipse(x, y, ax[i])
        ax[i].grid()
        ax[i].scatter(x, y, s=5)
        ax[i].set_title(titles[i])
    plt.suptitle("n = " + str(size))
    plt.show()


for size in sizes:
    for rho in rhos:
        pearson, spearman, quadrant = coeffs(normal, size, rho)
        results(pearson, spearman, quadrant)

    pearson, spearman, quadrant = coeffs(mix_normal, size, 0)
    results(pearson, spearman, quadrant)

    show_ellipse(size)