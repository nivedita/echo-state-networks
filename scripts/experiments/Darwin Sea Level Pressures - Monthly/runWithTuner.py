
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir, Tuner as tuner
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np


#Read data from the file
data = np.loadtxt('darwin.slp.txt')

#Input data and output data
nTraining = 1000
nTesting = 300
inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputData = data[1:nTraining+1].reshape((nTraining, 1))

#Tune the reservoir
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 100
initialTransient = 5
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

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Darwin Sea Level Pressure Prediction", "ESN with optimal config", "Time", "Sea Level Pressure")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testActualOutputData[:nTesting, 0])
outplot.setYSeries('Predicted Output', testPredictedOutputData[:nTesting, 0])
outplot.createOutput()







