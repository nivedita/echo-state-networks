from random import shuffle
import numpy as np
import scipy.special as spspec
import matplotlib.pyplot as plt
from plotting import OutputTimeSeries as outPlot
from random import shuffle

class GammaGenerator:
    def __init__(self, a, d, p):
        self.aParamList = a
        self.dParamList = d
        self.pParamList = p

        #Output plot
        self.plot = outPlot.OutputTimeSeriesPlot("Gammafunctions.html", "Gamma function", "with different parameters", "f(x)", "")

    def generate(self, X, fileName):
        f = open(fileName, 'w')
        result = []
        for a in self.aParamList:
            #for each value of d
            for d in self.dParamList:
                #for each value of p
                for p in self.pParamList:
                    #calulate the result
                    y = self.gengamma_pdf(X, a, d, p)
                    self.plot.setSeries("a = " + str(a) + ",d = " + str(d) + ",p = " + str(p), X, y)

                    #Generate the dataset
                    for i in range(X.shape[0]):
                        result.append(str(X[i]) + "," + str(y[i]) + "," + str(a) + "," + str(d) + "," + str(p) + "\n")

        #shuffle the result
        shuffle(result)

        #write to the data file
        for item in result:
            f.write(item)

        #return the plot object
        return self.plot

    def gengamma_pdf(self, x, a, d, p):
        '''PDF of the generalized Gamma distribution on [0, inf)
        Parameters: x: array like; input values
                    a: scale parameter
                    d,p: shape parameters
                    Returns: f: array like; results for values in x
        Asserts:    x >= 0
                    a, d, p > 0
        '''
        # if a <= 0 or d <= 0 or p <= 0:
        #     raise Exception, "arguments a, d, and p must be > 0"
        # if np.min(x) < 0:
        #     raise Exception, "all values in x must be >= 0"

        numer = p*x**(d-1.)*np.exp(-(x/a)**p)
        denom = a**d*spspec.gamma(d/p)
        f = numer / denom
        return f

if __name__ == "__main__":
    a = np.linspace(0.5,0.8,2)
    d = np.linspace(1.5, 1.8, 2)
    p = np.linspace(1.0, 1.3, 2)
    gg = GammaGenerator(a, d, p)
    x = np.linspace(0., 12., 250)
    gg.generate(x, "gamma.txt")

