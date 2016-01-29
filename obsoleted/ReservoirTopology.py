import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
        self.network = nx.erdos_renyi_graph(self.size, self.probability)
    # def generateConnectivityMatrix(self):
    #     random = np.random.rand(self.size, self.size)
    #
    #     connectivity = np.ones((self.size, self.size))
    #     randomIndices = random > self.probability
    #     connectivity[random > self.probability] = 0.0
    #     return connectivity, randomIndices

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        randomIndices = connectivity == 1
        return connectivity, randomIndices

class SmallWorldGraphs:
    def __init__(self, size, meanDegree, beta):
        self.size = size
        self.meanDegree = meanDegree
        self.beta = beta
        self.network = nx.newman_watts_strogatz_graph(self.size,self.meanDegree,self.beta) #No edges are removed in newman implementation (So, atleast we get a ring lattice)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        randomIndices = connectivity == 1.0
        return connectivity, randomIndices

class ScaleFreeNetworks:
    def __init__(self, size, attachmentCount):
        self.size = size
        self.m = attachmentCount
        self.network = nx.barabasi_albert_graph(self.size, self.m)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        randomIndices = connectivity == 1.0
        return connectivity, randomIndices


if __name__ == '__main__':
    scaleNw = ScaleFreeNetworks(10, 2)
    nx.draw_circular(scaleNw.network)
    plt.savefig("test.png")








