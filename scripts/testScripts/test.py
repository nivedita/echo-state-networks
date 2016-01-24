import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from plotting import NetworkPlot as networkPlot

# size = 5
# connectivity = 0.6
# connectivityMatrix = np.zeros((size, size))
#
# indices1 = []
# indices2 = []
# for i in range(size):
#     indices = np.random.choice(size, size=int(connectivity * size), replace=False)
#     connectivityMatrix[i, indices] = 1.0
#     for j in range(indices.shape[0]):
#         indices1.append(i)
#         indices2.append(indices[j])
#
# randomIndices = np.array(indices1), np.array(indices2)
# scalingMatrix = np.zeros((size, size))
# scalingMatrix[randomIndices] = 0.5
# print(scalingMatrix)
#
#
# indices = (connectivityMatrix == 1.0)
# scalingMatrixNew = np.zeros((size, size))
# scalingMatrixNew[indices] = 0.5
# print(scalingMatrix)
#
# a = np.array([1, 4, 7, 9, 10, 12, 14])
# depth = 2
# horizon = 1
# trainingVectors = []
#
# size = depth + horizon
# for i in range(depth+horizon):
#     # Lose the last few elements (for re-shaping)
#     loseCount =  a.shape[0] % (depth+horizon)
#     b = a[:a.shape[0]-loseCount]
#
#     # Reshape and form feature vectors
#     b = b.reshape((a.shape[0] / size, size)).tolist()
#     trainingVectors.extend(b)
#
#     # Move the array (like sliding window)
#     a = a[1:]
#
# # Separate the feature and target vectors
# trainingArray = np.array(trainingVectors)
# featureVectors = trainingArray[:,:depth]
# targetVectors = trainingArray[:,depth:]



# watts_strogatz = nx.newman_watts_strogatz_graph(5,4,0.2)
# connectivityMatrix = nx.to_numpy_matrix(watts_strogatz)
# plot1 = networkPlot.NetworkPlot("wattsmatrix.html", "1", "", connectivityMatrix)
# plot1.createOutput()
#
# connectivity = np.asarray(connectivityMatrix)
# plot2 = networkPlot.NetworkPlot("wattsarray.html", "2", "", connectivity)
# plot2.createOutput()
#
# difference = connectivityMatrix - connectivity
#
# nx.draw_circular(watts_strogatz)
# plt.savefig("test.png")



a = np.arange(10).reshape((5,2))
b = a[:3]
c = a[3:]
print(c)
