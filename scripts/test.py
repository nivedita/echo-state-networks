import numpy as np
size = 5
connectivity = 0.6
connectivityMatrix = np.zeros((size, size))

indices1 = []
indices2 = []
for i in range(size):
    indices = np.random.choice(size, size=int(connectivity * size), replace=False)
    connectivityMatrix[i, indices] = 1.0
    for j in range(indices.shape[0]):
        indices1.append(i)
        indices2.append(indices[j])

randomIndices = np.array(indices1), np.array(indices2)
scalingMatrix = np.zeros((size, size))
scalingMatrix[randomIndices] = 0.5
print(scalingMatrix)
