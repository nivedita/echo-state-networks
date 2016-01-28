import numpy as np
from timeit import default_timer as time
import threading
from multiprocessing import Process

def add(a, b):
    size = a.shape[0]
    c = np.zeros(size)
    for i in range(size):
        c[i] = a[i] + b[i]


def serialRun(times):
    N = 1000000
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    for i in range(times):
        add(a, b)


def threadRun(nthreads):
    N = 1000000
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    threads = []
    for i in range(nthreads):
        # Create each thread, passing it its chunk of numbers to factor
        # and output dict.
        t = Process(
                target=add,
                args=(a,b))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

ts = time()
serialRun(4)
te = time()
run_time = te - ts
print("The run time:"+str(run_time))

ts = time()
threadRun(4)
te = time()
run_time = te - ts
print("The thread run time:"+str(run_time))