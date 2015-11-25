#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)
from theano.tensor.tests.test_elemwise import test_IsInf_IsNan

from reservoir import Reservoir as reservoir
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
from datetime import datetime
import numpy as np
import os
from plotting import OutputPlot as outPlot
from gamma import Generator as generator

#Generate the gamma functions
a = np.linspace(0.5,0.8,2)
d = np.linspace(1.5, 1.8, 2)
p = np.linspace(1.0, 1.3, 2)
gg = generator.GammaGenerator(a, d, p)
x = np.linspace(0., 12., 250)
plot = gg.generate(x, "gamma.txt")

#Read data from the file
data = np.loadtxt("gamma.txt", delimiter=',')


#Set the number of points
nTraining = int(data.shape[0] * 0.9)
nTesting = 10
nTraining = data.shape[0] - 10

inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining, :2].reshape((nTraining, 2))))
outputData = data[:nTraining, 2:5].reshape((nTraining, 3))

#Train
res = reservoir.Reservoir(size = 500, spectralRadius = 1.05, inputScaling = 0.57, leakingRate = 0.30, initialTransient = 10, inputData = inputData, outputData = outputData)
res.trainReservoir()

#Test
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting, :2].reshape((nTesting, 2))))
testActualOutputData = data[nTraining:nTraining+nTesting, 2:5].reshape((nTesting, 3))
testPredictedOutputData = res.predict(testInputData)

test = res.predict(np.array([1, 0.530120481928,0.73282785323]).reshape((1, 3)))

#Plotting
for i in range(nTesting):
    name = "TP" + str(i+1)
    toolTipText ="Predicted: " + "a=" + str(testPredictedOutputData[i, 0]) + ",d=" + str(testPredictedOutputData[i, 1]) + ",p=" + str(testPredictedOutputData[i, 2]) + "<br>"
    toolTipText += "Actual: " + "a=" + str(testActualOutputData[i, 0]) + ",d=" + str(testActualOutputData[i, 1]) + ",p=" + str(testActualOutputData[i, 2])

    plot.setSingleDataPoint(name, str(testInputData[i, 1]), str(testInputData[i, 2]), toolTipText)

# toolTipText = "Actual: " + "a=" + str("0.8") + ",d=" + str("1.8") + ",p=" + str("1.3") + "<br>"
# toolTipText +="Predicted: " + "a=" + str(test[0, 0]) + ",d=" + str(test[0, 1]) + ",p=" + str(test[0, 2])
# plot.setSingleDataPoint("Test Point", 1.530120481928, 0.73282785323, toolTipText)
plot.createOutput()

print("Done!")