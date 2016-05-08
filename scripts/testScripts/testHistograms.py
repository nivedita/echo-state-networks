import numpy as np

X = [1,2,3,4,5,5,6, 6.5, 6,7, 8,9,10]

edges = [0,3,5,7,10]

hist = np.histogram(a=X, bins=5, density=True)

print(hist)