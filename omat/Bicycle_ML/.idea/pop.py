import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='population',
figsize=(12, 8), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# rng = np.random.RandomState(0)
# x = rng.randn(100)
# y = rng.randn(100)
# colors = rng.rand(100)
# sizes = 1000 * rng.rand(100)
#
# plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
# plt.colorbar()
# plt.show()