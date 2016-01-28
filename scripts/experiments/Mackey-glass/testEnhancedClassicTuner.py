#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import ClassicESN as reservoir, EnhancedClassicTuner as tuner
from plotting import OutputPlot as outputPlot
import numpy as np
from reservoir import Utility as util
from sklearn import preprocessing as pp
import os
from datetime import datetime

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data)

#Get only 4000 points
data = data[:5000].reshape((5000, 1))

# Split the data into training, validation and testing
trainingData, validationData, testingData = util.splitData(data, 0.4, 0.4, 0.2)
nValidation = validationData.shape[0]
nTesting = testingData.shape[0]

# Form feature vectors for training data
trainingInputData, trainingOutputData = util.formFeatureVectors(trainingData)

# Initial seed
initialSeedForValidation = trainingData[-1]
initialSeedForTesting = validationData[-1]

spectralRadiusBound = (0.5, 1.0)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.1, 0.5)
size = 256
initialTransient = 50
resTuner = tuner.ReservoirTuner(size=size,
                                initialTransient=initialTransient,
                                trainingInputData=trainingInputData,
                                trainingOutputData=trainingOutputData,
                                initialSeed=initialSeedForValidation,
                                validationOutputData=validationData,
                                spectralRadiusBound=spectralRadiusBound,
                                inputScalingBound=inputScalingBound,
                                reservoirScalingBound=reservoirScalingBound,
                                leakingRateBound=leakingRateBound)
spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightOptimum, reservoirWeightOptimum = resTuner.getOptimalParameters()

#Train
res = reservoir.Reservoir(size=size,
                          spectralRadius=spectralRadiusOptimum,
                          inputScaling=inputScalingOptimum,
                          reservoirScaling=reservoirScalingOptimum,
                          leakingRate=leakingRateOptimum,
                          initialTransient=initialTransient,
                          inputData=trainingInputData,
                          outputData=trainingOutputData,
                          inputWeightRandom=inputWeightOptimum,
                          reservoirWeightRandom=reservoirWeightOptimum)
res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(trainingInputData)


predictedValidationOutputData = util.predictFuture(res, initialSeedForValidation, nValidation)
predictedTestOutputData = util.predictFuture(res, initialSeedForTesting, nTesting)


#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/PredictionValidation.html", "Mackey-Glass Time Series", "Prediction on validation set", "Time", "Output")
outplot.setXSeries(np.arange(1, nValidation + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(validationData[:nValidation, 0]))
outplot.setYSeries('Predicted Output', minMax.inverse_transform(predictedValidationOutputData[:nValidation, 0]))
outplot.createOutput()
print("Done!")

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot(outputFolderName + "/PredictionTesting.html", "Mackey-Glass Time Series", "Prediction on Testing", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(testingData[:nTesting, 0]))
outplot.setYSeries('Predicted Output', minMax.inverse_transform(predictedTestOutputData[:nTesting, 0]))
outplot.createOutput()
print("Done!")