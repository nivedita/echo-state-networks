import numpy as np
from scipy.special import expit

class HyperbolicTangent(object):
    def __call__(self, x):
        return np.tanh(x)


class SigmoidFunction(object):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x):
        return expit(self.beta * x)


class ReLU(object):
    def __call__(self, x):
        np.maximum(x, np.zeros(x.shape))