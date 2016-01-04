import numpy as np

class RandomTopology:
    def __init__(self, size, connectivity):
        self.size = size
        self.connectivity = connectivity

    def generateConnectivityMatrix(self):
        #Initialize the matrix to zeros
        connectivity = np.zeros((self.size, self.size))

        indices1 = []
        indices2 = []
        for i in range(self.size):
            indices = np.random.choice(self.size, size=int(self.connectivity * self.size), replace=False)
            connectivity[i, indices] = 1.0
            for j in range(indices.shape[0]):
                indices1.append(i)
                indices2.append(indices[j])
        randomIndices = np.array(indices1), np.array(indices2)
        return connectivity, randomIndices
