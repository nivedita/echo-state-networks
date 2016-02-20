import numpy as np
x = np.arange(0,9)
print(x)
indices = np.where(x % 3 == 1)
print(indices)

print(x[indices])





