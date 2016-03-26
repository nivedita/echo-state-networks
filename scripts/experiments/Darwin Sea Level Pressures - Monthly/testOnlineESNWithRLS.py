#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import onlineESNWithRLS as ESN, ReservoirTopology as topology
from plotting import OutputPlot as outputPlot
import numpy as np
import os
from datetime import datetime
from sklearn import preprocessing as pp
from reservoir import Utility as util
from performance import ErrorMetrics as rmse

# Read data from the file
data = np.loadtxt('darwin.slp.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data).reshape((data.shape[0],1))

# Divide the data into training data and testing data
trainingData, testingData = util.splitData2(data, 0.8)
nTesting = testingData.shape[0]

# Form feature vectors
inputTrainingData, outputTrainingData = util.formFeatureVectors(trainingData)

# Tune the network
size = int(trainingData.shape[0]/10)
initialTransient = 100

# Input-to-reservoir fully connected
inputWeight = topology.ClassicInputTopology(inputSize=inputTrainingData.shape[1], reservoirSize=size).generateWeightMatrix()

# Reservoir-to-reservoir fully connected
reservoirWeight = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()

res = ESN.Reservoir(size=size,
                    inputData=inputTrainingData,
                    outputData=outputTrainingData,
                    spectralRadius=0.79,
                    inputScaling=0.5,
                    reservoirScaling=0.5,
                    leakingRate=0.3,
                    initialTransient=initialTransient,
                    inputWeightRandom=inputWeight,
                    reservoirWeightRandom=reservoirWeight)
res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(inputTrainingData[-initialTransient:])

#Predict future values
predictedTestOutputData = util.predictFuture(res, trainingData[-1], nTesting)

#Calculate the error
errorFunction = rmse.MeanSquareError()
error = errorFunction.compute(testingData, predictedTestOutputData)
print("Regression error:"+str(error))


#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series - Classic ESN", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(testingData[:nTesting, 0]))
outplot.setYSeries('Predicted Output', minMax.inverse_transform(predictedTestOutputData[:nTesting, 0]))
outplot.createOutput()
print("Done!")