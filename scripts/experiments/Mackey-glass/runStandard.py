#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import EchoStateNetwork as ESN, Tuner as tuner, ReservoirTopology as topology
from plotting import OutputPlot as outputPlot
import numpy as np
import os
from datetime import datetime
from sklearn import preprocessing as pp

# Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((0,1))
data = minMax.fit_transform(data)

# Training Input and Output data
nTraining = 2000
nTesting = 2000
trainingInputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
trainingOutputData = data[1:nTraining+1].reshape((nTraining, 1))

# Testing Input and Output data
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))

# Tune the network
size = 256
initialTransient = 50
topology = topology.RandomTopology(size=size, connectivity=0.9)
inputConnectivity = 0.7
spectralRadiusBound = (0.0, 1.0)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
resTuner = tuner.ESNTuner(size=size,
                          initialTransient=initialTransient,
                          trainingInputData=trainingInputData,
                          trainingOutputData=trainingOutputData,
                          validationInputData=trainingInputData,
                          validationOutputData=trainingOutputData,
                          spectralRadiusBound=spectralRadiusBound,
                          inputScalingBound=inputScalingBound,
                          reservoirScalingBound=reservoirScalingBound,
                          leakingRateBound=leakingRateBound,
                          reservoirTopology=topology,
                          inputConnectivity=inputConnectivity)

spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightConn, reservoirWeightConn = resTuner.getOptimalParameters()

#Train the reservoir with optimal parameters
size = 256
res = ESN.EchoStateNetwork(size=size,
                           inputData=trainingInputData,
                           outputData=trainingOutputData,
                           reservoirTopology=topology,
                           spectralRadius=spectralRadiusOptimum,
                           inputScaling=inputScalingOptimum,
                           reservoirScaling=reservoirScalingOptimum,
                           leakingRate=leakingRateOptimum,
                           initialTransient=initialTransient,
                           inputWeightConn=inputWeightConn,
                           reservoirWeightConn=reservoirWeightConn)

res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(trainingInputData)


#Predict future values
predictedTestOutputData = []
lastAvailableData = testInputData[0, 1]
for i in range(nTesting):
    query = [1.0]
    query.append(lastAvailableData)

    #Predict the next point
    nextPoint = res.predict(np.array(query).reshape((1,2)))[0,0]
    predictedTestOutputData.append(nextPoint)

    lastAvailableData = nextPoint

predictedTestOutputData = np.array(predictedTestOutputData).reshape((nTesting, 1))

#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs-" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(testActualOutputData[:nTesting, 0]))
outplot.setYSeries('Predicted Output', minMax.inverse_transform(predictedTestOutputData[:nTesting, 0]))
outplot.createOutput()
print("Done!")