# 初期情報として与えられたw', h'により更新された事後分布を求める
import matplotlib.pyplot as plt
import numpy as np

xmin = 0
xmax = 120000
num_plots = 4
rows = 2
fix, axes = plt.subplots(rows, (num_plots + rows - 1) // rows, figsize=(15, 10))
axes = axes.flatten()

# 事前分布
mean = 65000
std = 21280
var = std**2

for i, sigma in enumerate(range(1000, 10001, 3000)):
    ax = axes[i]
    sigma2 = sigma**2

    for w in range(xmin, xmax + 1, 20000):
        post_mean = (w * var + mean * sigma2) / (var + sigma2)
        post_var = var * sigma2 / (var + sigma2)
        post_std = np.sqrt(post_var)
        print(f"sigma = {sigma}, w={w}, post_mean={post_mean:.1f}, post_std={post_std:.1f}")

        x = np.linspace(xmin, xmax, 1000)
        y = np.exp(-((x - post_mean) ** 2) / (2 * post_var)) / np.sqrt(
            2 * np.pi * post_var
        )
        ax.plot(x, y, label=f"w={w}")

    ax.set_title(f"sigma = {sigma}")
    ax.grid(True)

plt.tight_layout()
plt.show()
