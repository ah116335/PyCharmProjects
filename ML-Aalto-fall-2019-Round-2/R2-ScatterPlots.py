from matplotlib import pyplot as plt

X, y = GetFeaturesLabels(10, 10)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].scatter(X[:, 0], y)
axs[0].set_title('average number of rooms per dwelling vs. price')
axs[0].set_xlabel(r'feature $x_{1}$')
axs[0].set_ylabel('house price $y$')

axs[1].scatter(X[:, 1], y)
axs[1].set_xlabel(r'feature $x_{2}$')
axs[1].set_title('nitric oxide level vs. price')
axs[1].set_ylabel('house price $y$')

plt.show()

