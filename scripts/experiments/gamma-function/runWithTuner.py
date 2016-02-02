#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)
from theano.tensor.tests.test_elemwise import test_IsInf_IsNan

from reservoir import EchoStateNetwork as reservoir, Tuner as tuner
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import ErrorMetrics as rmse
from datetime import datetime
import numpy as np
import os
from plotting import OutputPlot as outPlot
from gamma import Generator as generator

#Generate the gamma functions
a = np.linspace(0.5,2.5,5)
d = np.linspace(2.0, 4.0, 5)
p = np.linspace(1.0, 3.0, 5)
gg = generator.GammaGenerator(a, d, p)
x = np.linspace(0., 6., 250)
plot = gg.generate(x, "gamma.txt")

#Read data from the file
data = np.loadtxt("gamma.txt", delimiter=',')


#Set the number of points
nTraining = int(data.shape[0] * 0.9)
nTesting = 10
nTraining = data.shape[0] - nTesting

inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining, :2].reshape((nTraining, 2))))
outputData = data[:nTraining, 2:5].reshape((nTraining, 3))


#Tune the reservoir
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting, :2].reshape((nTesting, 2))))
testActualOutputData = data[nTraining:nTraining+nTesting, 2:5].reshape((nTesting, 3))
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 100
initialTransient = 10
resTuner = tuner.ReservoirTuner(size = size,
                                initialTransient=initialTransient,
                                trainingInputData=inputData,
                                trainingOutputData=outputData,
                                validationInputData=testInputData,
                                validationOutputData=testActualOutputData,
                                spectralRadiusBound=spectralRadiusBound,
                                inputScalingBound=inputScalingBound,
                                leakingRateBound=leakingRateBound)
spectralRadiusOptimum, inputScalingOptimum, leakingRateOptimum, inputWeightOptimum, reservoirWeightOptimum = resTuner.getOptimalParameters()

#Train the reservoir with the optimal parameters
res = reservoir.Reservoir(size=size,
                          spectralRadius=spectralRadiusOptimum,
                          inputScaling=inputScalingOptimum,
                          leakingRate=leakingRateOptimum,
                          initialTransient=initialTransient,
                          inputData=inputData,
                          outputData=outputData,
                          inputWeightRandom=inputWeightOptimum,
                          reservoirWeightRandom=reservoirWeightOptimum)
res.trainReservoir()

#Predict
testPredictedOutputData = res.predict(testInputData)

#Plotting
for i in range(nTesting):
    toolTipText ="Predicted: " + "a=" + str(testPredictedOutputData[i, 0]) + ",d=" + str(testPredictedOutputData[i, 1]) + ",p=" + str(testPredictedOutputData[i, 2]) + "<br>"
    toolTipText += "Actual: " + "a=" + str(testActualOutputData[i, 0]) + ",d=" + str(testActualOutputData[i, 1]) + ",p=" + str(testActualOutputData[i, 2])
    name = "Test Point:" + str(i+1) + " "+ toolTipText

    plot.setSeries(name, np.array([testInputData[i, 1]]), np.array([testInputData[i, 2]]))

plot.createOutput()
print("Done!")