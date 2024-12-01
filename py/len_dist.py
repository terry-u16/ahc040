# 箱の辺の長さの事前分布を正規分布で近似する
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

N = 1000000
x = []

for i in range(N):
    U = 100000
    L = random.randint(U // 10, U // 2)
    x.append(random.randint(L, U))

x = np.array(x)
mean = np.mean(x)
var = np.var(x)
std = np.std(x)
skew = stats.skew(x)
kurtosis = stats.kurtosis(x)

print("mean               :", mean)
print("variance           :", var)
print("standard deviation :", std)
print("skewness           :", skew)
print("kurtosis           :", kurtosis)

mu, std = stats.norm.fit(x)
print(mu, std)

plt.hist(x, bins=100, density=True, alpha=0.6)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, "k", linewidth=2)

plt.show()
