import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.optimize as opt


def standard(x):
    return 2 + 2 * x


def standard_with_error(x):
    y = []
    for element in x:
        y.append(standard(element) + stats.norm.rvs(0, 1))
    return y


def mnk_params(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def mnk(x, y):
    beta_0, beta_1 = mnk_params(x, y)
    print("a = " + str(beta_0) + ", b = " + str(beta_1))
    return [beta_0 + beta_1 * element for element in x]


def mnm_min(params, x, y):
    beta_0, beta_1 = params
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - beta_0 - beta_1 * x[i])
    return sum


def mnm_params(x, y):
    beta_0, beta_1 = mnk_params(x, y)
    optimum = opt.minimize(mnm_min, [beta_0, beta_1], args=(x, y), method="SLSQP")
    return optimum.x[0], optimum.x[1]


def mnm(x, y):
    beta_0, beta_1 = mnm_params(x, y)
    print("a = " + str(beta_0) + ", b = " + str(beta_1))
    return [beta_0 + beta_1 * element for element in x]


def plot_regression(x, y, name):
    y_mnk = mnk(x, y)
    y_mnm = mnm(x, y)

    plt.plot(x, standard(x), color="red", label="Модель")
    plt.plot(x, y_mnk, color="green", label="МНК")
    plt.plot(x, y_mnm, color="orange", label="МНМ")
    plt.scatter(x, y, c="blue", label="Выборка")
    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    plt.title(name)
    plt.show()


x = np.arange(-1.8, 2, 0.2)
y = standard_with_error(x)
plot_regression(x, y, "Распределение без возмущения")
y[0] += 10
y[-1] -= 10
plot_regression(x, y, "Распределение с возмущением")