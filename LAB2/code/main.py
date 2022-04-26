from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy as np
import math

sizes = [10, 100, 1000]
repetitions = 1000
rounding = 6


def z_r(sample, size):
    return (sample[0] + sample[size - 1]) / 2


def z_p(sample, n_p):
    if n_p.is_integer():
        return sample[int(n_p)]
    else:
        return sample[int(n_p) + 1]


def z_q(sample, size):
    return (z_p(sample, size / 4) + z_p(sample, 3 * size / 4)) / 2


def z_tr(sample, size):
    r = int(size / 4)
    amount = 0
    for i in range(r + 1, size - r + 1):
        amount += sample[i]
    return (1 / (size - 2 * r)) * amount


def build_params(distribution):
    for size in sizes:
        mean_list, med_list, z_r_list, z_q_list, z_tr_list, e_list, d_list = [], [], [], [], [], [], []
        lists = [mean_list, med_list, z_r_list, z_q_list, z_tr_list]

        for i in range(repetitions):
            sample = distribution.rvs(size=size)
            sample.sort()
            mean_list.append(np.mean(sample))
            med_list.append(np.median(sample))
            z_r_list.append(z_r(sample, size))
            z_q_list.append(z_q(sample, size))
            z_tr_list.append(z_tr(sample, size))

        for part in lists:
            e_list.append(round(np.mean(part), rounding))
            d_list.append(round(np.std(part) ** 2, rounding))
           
        print(e_list)
        print(d_list)


print("-----------")
build_params(norm())
print("-----------")
build_params(cauchy())
print("-----------")
build_params(laplace(loc=0, scale=1 / math.sqrt(2)))
print("-----------")
build_params(poisson(10))
print("-----------")
build_params(uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)))