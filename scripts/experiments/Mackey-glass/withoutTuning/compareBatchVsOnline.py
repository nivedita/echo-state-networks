#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import classicESN as ESN, onlineESNWithRLS as onlineESN, ReservoirTopology as topology
from plotting import OutputPlot as outputPlot
import numpy as np
import os
from datetime import datetime
from sklearn import preprocessing as pp
from reservoir import Utility as util
from performance import ErrorMetrics as rmse

# Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data)

#Get only 6000 points
data = data[:5000].reshape((5000, 1))

# Number of points - 5000
trainingData, testingData = util.splitData2(data, 0.4)
nTesting = testingData.shape[0]

# Form feature vectors
inputTrainingData, outputTrainingData = util.formFeatureVectors(trainingData)

# Tune the network
size = 256
initialTransient = 50

# Input-to-reservoir fully connected
inputWeight = topology.ClassicInputTopology(inputSize=inputTrainingData.shape[1], reservoirSize=size).generateWeightMatrix()

# Reservoir-to-reservoir fully connected
reservoirWeight = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()

resBatch = ESN.Reservoir(size=size,
                    inputData=inputTrainingData,
                    outputData=outputTrainingData,
                    spectralRadius=0.79,
                    inputScaling=0.5,
                    reservoirScaling=0.5,
                    leakingRate=0.3,
                    initialTransient=initialTransient,
                    inputWeightRandom=inputWeight,
                    reservoirWeightRandom=reservoirWeight)
resBatch.trainReservoir()

resOnline = onlineESN.Reservoir(size=size,
                    inputData=inputTrainingData,
                    outputData=outputTrainingData,
                    spectralRadius=0.79,
                    inputScaling=0.5,
                    reservoirScaling=0.5,
                    leakingRate=0.3,
                    initialTransient=initialTransient,
                    inputWeightRandom=inputWeight,
                    reservoirWeightRandom=reservoirWeight)
resOnline.trainReservoir()

#Warm up
predictedTrainingOutputData = resBatch.predict(inputTrainingData[-initialTransient:])
predictedTrainingOutputData = resOnline.predict(inputTrainingData[-initialTransient:])

#Predict future values
predictedTestOutputDataBatch = util.predictFuture(resBatch, trainingData[-1], nTesting)
predictedTestOutputDataOnline = util.predictFuture(resOnline, trainingData[-1], nTesting)

#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series - Comparison of Batch vs Online", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(testingData[:nTesting, 0]))
outplot.setYSeries('Predicted Batch learning Output', minMax.inverse_transform(predictedTestOutputDataBatch[:nTesting, 0]))
outplot.setYSeries('Predicted Online RLS learning Output', minMax.inverse_transform(predictedTestOutputDataOnline[:nTesting, 0]))
outplot.createOutput()
print("Done!")