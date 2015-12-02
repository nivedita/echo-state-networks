from scipy import optimize
import numpy as np

def fun(x0):
    x1 = x0[0]
    x2 = x0[1]
    value = 2 * x1-x2
    return abs(value)


if __name__ == '__main__':
    minimizer_kwargs = {"method": "BFGS"}
    ret = optimize.differential_evolution(fun, bounds = [(0,5),(0,5)])
    print (ret)