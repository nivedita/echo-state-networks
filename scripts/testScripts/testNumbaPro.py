from __future__ import print_function
import numpy as np
from numba import *
from timeit import default_timer as time

d = np.ones(1)
@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cuda_sum(a, b, c):
    print(d)
    i = cuda.grid(1)
    c[i] = a[i] + b[i] + d[0]

def cpu_sum(a, b):
    size = a.shape[0]
    c = np.zeros(size)
    for i in range(size):
            c[i] = a[i] + b[i]
    return c


griddim = 10000, 1
blockdim = 32, 1, 1
N = griddim[0] * blockdim[0]
print("N", N)
cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.empty_like(a)

ts = time()
cuda_sum_configured(a, b, c)
te = time()
cuda_time = te - ts
print("The cuda time: "+ str(cuda_time))

ts = time()
c = cpu_sum(a, b)
te = time()
cpu_time = te-ts
print("The CPU time: "+str(cpu_time))

print("The speed factor for CUDA is:"+ str(cpu_time/cuda_time))