import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)

plt.scatter(x, y, alpha=0.5)
plt.show()