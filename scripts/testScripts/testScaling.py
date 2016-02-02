import numpy as np
from reservoir import ReservoirTopology as topology

conn = topology.RandomTopology(size=5, connectivity=0.6).generateConnectivityMatrix()
weight = np.random.rand(5,5)
weight = conn * weight
scaling = 0.5
weight[weight != 0.0] = weight[weight != 0.0] - 0.5
print(weight)
