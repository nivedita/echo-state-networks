import numpy as np

class RandomTopology:
    def __init__(self, size, connectivity):
        self.size = size
        self.connectivity = connectivity

    def generateConnectivityMatrix(self):
        #Initialize the matrix to zeros
        connectivity = np.zeros((self.size, self.size))
        for i in range(self.size):
            indices = np.random.choice(self.size, size=int(self.connectivity * self.size), replace=False)
            connectivity[i, indices] = 1.0
        randomIndices = connectivity == 1.0
        return connectivity, randomIndices


class ErdosRenyiTopology:
    def __init__(self, size, probability):
        self.size = size
        self.probability = probability
        self.expectedNumberOfLinks = None
        self.averageDegree = None
        self.clusteringCoefficient = None

    def generateConnectivityMatrix(self):
        random = np.random.rand(self.size, self.size)

        connectivity = np.ones((self.size, self.size))
        randomIndices = random > self.probability
        connectivity[random > self.probability] = 0.0
        return connectivity, randomIndices

    def calculateNetworkParameters(self):

        #Expected number of links
        self.expectedNumberOfLinks = self.probability * self.size * (self.size - 1) / 2

        return self.expectedNumberOfLinks







