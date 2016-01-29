
from reservoir import ReservoirTopology as toplogy
import numpy as np
reservoirConnMatrix = toplogy.RandomTopology(size=2, connectivity=0.5).generateConnectivityMatrix()
reservoirWeightRandom = np.arange(4).reshape((2,2))

multiplyOperator = reservoirWeightRandom * reservoirConnMatrix
multiplyFunction = np.multiply(reservoirWeightRandom, reservoirConnMatrix)

print(reservoirWeightRandom)