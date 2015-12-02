#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir, Tuner as tuner
from plotting import OutputTimeSeries as outTimePlot
from datetime import datetime
import numpy as np
import os
from timeseries import TimeSeries as ts
from sklearn import preprocessing as pp

# Read data from the file
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')

# #Normalize the raw data
minMax = pp.MinMaxScaler((-1,+1))
rawData[:, 4] = minMax.fit_transform(rawData[:, 4])

# Set the cut off point for training
# For instance, let's test the horizon points
# So, split last horizon points at the end
depth = 200
horizon = 30
valueIndex = 4

#Pre-Process the data to form feature and target vectors
tsp = ts.TimeSeriesProcessor(rawData[:, valueIndex], depth, horizon)
processedData = tsp.getProcessedData()

#Append bias
processedData = np.hstack((np.ones((processedData.shape[0], 1)),processedData))

#Divide the data into training, validation and testing
cutOffIndexTesting = processedData.shape[0] - 1
cutOffIndexValidation = int(cutOffIndexTesting * 0.7)

#Training data
# trainingInputData = processedData[:cutOffIndexValidation,:1+depth]
# trainingOutputData = processedData[:cutOffIndexValidation,1+depth:1+depth+horizon]

trainingInputData = processedData[:cutOffIndexTesting,:1+depth]
trainingOutputData = processedData[:cutOffIndexTesting,1+depth:1+depth+horizon]


#Validation data
# validationInputData = processedData[cutOffIndexValidation:cutOffIndexTesting,:1+depth]
# validationOutputData = processedData[cutOffIndexValidation:cutOffIndexTesting,1+depth:1+depth+horizon]
validationInputData = trainingInputData
validationOutputData = trainingOutputData

#Testing data
testingInputData = processedData[cutOffIndexTesting:,:1+depth]
testingOutputData = processedData[cutOffIndexTesting:,1+depth:1+depth+horizon]

#Tune the reservoir
spectralRadiusBound = (0.0, 1.9)
inputScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 200
initialTransient = 50
resTuner = tuner.ReservoirTuner(size=size,
                                 initialTransient=initialTransient,
                                 trainingInputData=trainingInputData,
                                 trainingOutputData=trainingOutputData,
                                 validationInputData=validationInputData,
                                 validationOutputData=validationOutputData,
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
                        inputData=trainingInputData,
                        outputData=trainingOutputData,
                        inputWeightRandom=inputWeightOptimum,
                        reservoirWeightRandom=reservoirWeightOptimum)

res.trainReservoir()

#Warm u
trainingPred = res.predict(trainingInputData)

#Predict for test data
testingPredictedOutputData = res.predict(testingInputData)
testActualOutputData = minMax.inverse_transform(testingOutputData.reshape((horizon, 1))[ :,0])
testPredictedOutputData = minMax.inverse_transform(testingPredictedOutputData.reshape((horizon, 1))[ :,0])

#Create the needed lists for plotting
xAxis = []
index = rawData.shape[0] - horizon
while index < rawData.shape[0]:
    day = "Date.UTC(" + str(int(rawData[index, 0])) +","+ str(int(rawData[index, 1]) -1) + "," + str(int(rawData[index, 2])) +")"
    xAxis.append(day)
    index += 1

# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxis), testActualOutputData)
outplot.setSeries('Predicted Output', np.array(xAxis), testPredictedOutputData)
outplot.createOutput()

print ("Done!")