import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class ClassicReservoirTopology:
    def __init__(self, size):
        self.size = size

    def generateWeightMatrix(self):
        reservoirWeightRandom = np.random.rand(self.size, self.size)
        return reservoirWeightRandom

class ClassicInputTopology:
    def __init__(self, inputSize, reservoirSize):
        self.inputSize = inputSize
        self.reservoirSize = reservoirSize

    def generateWeightMatrix(self):
        inputWeightRandom = np.random.rand(self.reservoirSize, self.inputSize)
        return inputWeightRandom

class RandomInputTopology:
    def __init__(self, inputSize, reservoirSize, inputConnectivity):
        self.inputSize = inputSize
        self.reservoirSize = reservoirSize
        self.inputConnectivity = inputConnectivity

    def generateConnectivityMatrix(self):
        connectivity = np.zeros((self.reservoirSize, self.inputSize))
        for i in range(self.reservoirSize):
            indices = np.random.choice(self.inputSize, size=int(self.inputConnectivity * self.inputSize), replace=False)
            connectivity[i, indices] = 1.0
        return connectivity

    def generateWeightMatrix(self):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.rand(self.reservoirSize, self.inputSize)
        weight = random * self.generateConnectivityMatrix()
        return weight

class RandomReservoirTopology:
    def __init__(self, size, connectivity):
        self.size = size
        self.connectivity = connectivity

    def generateConnectivityMatrix(self):
        #Initialize the matrix to zeros
        connectivity = np.zeros((self.size, self.size))
        for i in range(self.size):
            indices = np.random.choice(self.size, size=int(self.connectivity * self.size), replace=False)
            connectivity[i, indices] = 1.0
        return connectivity

    def generateWeightMatrix(self):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.rand(self.size, self.size)
        weight = random * self.generateConnectivityMatrix()
        return weight

class ErdosRenyiTopology:
    def __init__(self, size, probability):
        self.size = size
        self.probability = probability
        self.network = nx.erdos_renyi_graph(self.size, self.probability)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        return connectivity

    def generateWeightMatrix(self):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.rand(self.size, self.size)
        weight = random * self.generateConnectivityMatrix()
        return weight

class SmallWorldGraphs:
    def __init__(self, size, meanDegree, beta):
        self.size = size
        self.meanDegree = meanDegree
        self.beta = beta
        self.network = nx.newman_watts_strogatz_graph(self.size,self.meanDegree,self.beta) #No edges are removed in newman implementation (So, atleast we get a ring lattice)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        return connectivity

    def generateWeightMatrix(self):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.rand(self.size, self.size)
        weight = random * self.generateConnectivityMatrix()
        return weight

class ScaleFreeNetworks:
    def __init__(self, size, attachmentCount):
        self.size = size
        self.m = attachmentCount
        self.network = nx.barabasi_albert_graph(self.size, self.m)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        return connectivity

    def generateWeightMatrix(self):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.rand(self.size, self.size)
        weight = random * self.generateConnectivityMatrix()
        return weight


if __name__ == '__main__':
    scaleNw = ScaleFreeNetworks(10, 2)
    nx.draw_circular(scaleNw.network)
    plt.savefig("test.png")








