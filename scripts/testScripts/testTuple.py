
import numpy as np
a = tuple(np.array([1,2,3]).tolist())
b = (1,2,3)
c = (1,4,3)

if(a == b):
    print("A and B are same")
else:
    print("A and B are not same")

if(a == c):
    print("A and C are same")
else:
    print("A and C are not same")

pop = []
pop.append((b, 100))
print(pop[0][0])