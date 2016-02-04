import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy import linalg as la

def least_square_solve(Y, X):
    """

    :param Y: N x 1
    :param X_transpose: N X D
    """

    X_transpose = X
    X = X_transpose.T

    # Solve for W in Y = X_transpose * W
    # ie. W = Y (X_transpose * X)`T  * X_transpose

    term1 = la.inv(np.dot(X_transpose,X))
    term2 = np.dot(term1,X_transpose)
    term3 = np.dot(Y, term2)
    return term3

def data_matrix_V1(x):
    n = len(x)
    return np.vstack((x, np.ones(n))).T

def lsq_solution_V1(X, y):
    w = np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), y)
    return w

def create_data(n, xmin=-2, xmax=12, a=1.1, b=2.0):
    x = rnd.random(n) * (xmax-xmin) + xmin
    y = a * x + b + rnd.randn(n) * 0.5
    return x, y

if __name__ == '__main__':
    n = 25
    x, y = create_data(n)
    X = data_matrix_V1(x)
    w = lsq_solution_V1(X, y)[0]
    yhat = np.dot(X, w)
    plt.plot(x, yhat,"-")
    plt.plot(x, y,"o")
    plt.show()