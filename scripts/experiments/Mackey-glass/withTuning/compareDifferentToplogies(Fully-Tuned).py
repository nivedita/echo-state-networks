#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

import sys
sys.path.append('C:/Users/Raj/PycharmProjects/echo-state-networks')

from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
import numpy as np
from reservoir import Utility as util
from sklearn import preprocessing as pp
import os
from datetime import datetime
from timeit import default_timer as time
from performance import ErrorMetrics as metrics
from reservoir import ReservoirTopology as topology

outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)

startTime = time()

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data)

#Get only 6000 points
data = data[:6000].reshape((6000, 1))

# Split the data into training, validation and testing
trainingData, validationData, testingData = util.splitData(data, 0.5, 0.25, 0.25)
nValidation = validationData.shape[0]
nTesting = testingData.shape[0]

# Form feature vectors for training data
trainingInputData, trainingOutputData = util.formFeatureVectors(trainingData)
actualOutputData = minMax.inverse_transform(testingData)[:,0]

# Initial seed
initialSeedForValidation = trainingData[-1]

# Error function
errorFun = metrics.MeanSquareError()

# Number of iterations
iterations = 1
reservoirSize = 256

# Run Classic ESN
classicESNError = 0
print("\n Running Classic ESN Tuner..")
for i in range(iterations):
    predictedOutputData, error = util.tuneTrainPredict(trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            validationOutputData=validationData,
                                            initialInputSeedForValidation=initialSeedForValidation,
                                            reservoirTopology=topology.ClassicReservoirTopology(size=reservoirSize),
                                            testingData=testingData
                                            )
    classicESNPredicted = minMax.inverse_transform(predictedOutputData)
    classicESNError += error
classicESNError = classicESNError/iterations

# Run random ESN Tuner
randomESNError = 0
connectivityOptimum = 0.72999999999999998
print("\n Running Random ESN Tuner..")
for i in range(iterations):
    predictedOutputData, error = util.tuneTrainPredict(trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            validationOutputData=validationData,
                                            initialInputSeedForValidation=initialSeedForValidation,
                                            reservoirTopology=topology.RandomReservoirTopology(size=reservoirSize, connectivity=connectivityOptimum),
                                            testingData=testingData)
    randomESNPredicted = minMax.inverse_transform(predictedOutputData)
    randomESNError += error
randomESNError = randomESNError/iterations

# Run Erdos ESN Tuner
erdosESNError = 0
probabilityOptimum = 1.0
print("\n Running Erdos ESN Tuner..")
for i in range(iterations):
    print("\n Iteration:"+str(i+1))
    predictedOutputData, error = util.tuneTrainPredict(trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            validationOutputData=validationData,
                                            initialInputSeedForValidation=initialSeedForValidation,
                                            reservoirTopology=topology.ErdosRenyiTopology(size=reservoirSize, probability=probabilityOptimum),
                                            testingData=testingData
                                            )
    erdosPredictedOutput = minMax.inverse_transform(predictedOutputData)
    erdosESNError += error
erdosESNError = erdosESNError/iterations


# Run Small World Graphs ESN Tuner
smallWorldESNError = 0
meanDegreeOptimum = int(188.0)
betaOptimum = 0.46000000000000002
print("\n Running Small World Graphs ESN Tuner..")
for i in range(iterations):
    predictedOutputData, error = util.tuneTrainPredict(trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            validationOutputData=validationData,
                                            initialInputSeedForValidation=initialSeedForValidation,
                                            reservoirTopology=topology.SmallWorldGraphs(size=reservoirSize, meanDegree=meanDegreeOptimum, beta=betaOptimum),
                                            testingData=testingData
                                            )
    smallWorldPredictedOutput = minMax.inverse_transform(predictedOutputData)
    smallWorldESNError += error
smallWorldESNError = smallWorldESNError/iterations

# Run Scale Free Networks ESN Tuner
scaleFreeESNError = 0
attachmentOptimum = int(68)
print("\n Running Scale Free Networks ESN Tuner..")
for i in range(iterations):
    predictedOutputData, error = util.tuneTrainPredict(trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            validationOutputData=validationData,
                                            initialInputSeedForValidation=initialSeedForValidation,
                                            reservoirTopology=topology.ScaleFreeNetworks(size=reservoirSize, attachmentCount=attachmentOptimum),
                                            testingData=testingData)
    scaleFreePredictedOutput = minMax.inverse_transform(predictedOutputData)
    scaleFreeESNError += error
scaleFreeESNError = scaleFreeESNError/iterations

# Plotting of regression error
errPlot = errorPlot.ErrorPlot(outputFolderName + "/RegressionError.html", "Comparison of different graph topologies", "with connectivity parameters tuned", "ESN Configuration", "Total Error")
errPlot.setXAxis(np.array(['Classic ESN', 'Random ESN', 'Erdos Renyi ESN', 'Small World Graph ESN', 'Scale Free Network ESN']))
errPlot.setYAxis('Mean Squared Error', np.array([classicESNError, randomESNError, erdosESNError, smallWorldESNError, scaleFreeESNError]))
errPlot.createOutput()

#Plotting of the prediction output
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series - Differential Evolution Optimization", "Predicted vs Actual", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', actualOutputData)
outplot.setYSeries('Classic ESN Output', classicESNPredicted)
outplot.setYSeries('Random ESN Output', randomESNPredicted)
outplot.setYSeries('Erdos ESN Output', erdosPredictedOutput)
outplot.setYSeries('Small World ESN Output', smallWorldPredictedOutput)
outplot.setYSeries('Scale ESN Output', scaleFreePredictedOutput)
outplot.createOutput()
