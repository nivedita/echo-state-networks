import numpy as np

arr = np.arange(9).reshape((3, 3))

print(arr)
arr = np.transpose(arr)
np.random.shuffle(arr)

print(arr)

size = 5
connectivity = 0.6

conn = np.zeros(size)
indices = np.random.choice(size, size=int(connectivity * size), replace=False)
conn[indices] = 1.0

conn = np.tile(conn, size).reshape((size, size))
transposed = np.transpose(conn)
np.random.shuffle(transposed)
print(conn)
