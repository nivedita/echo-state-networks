#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir, Tuner as tuner
from plotting import OutputTimeSeries as outTimePlot
from datetime import date, timedelta, datetime
import numpy as np
import os
from timeseries import TimeSeries as ts
from sklearn import preprocessing as pp

# Depth and Horizon
depth = 100
horizon = 30
valueIndex = 4

# Read data from the file
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')

# Read until the horizon
actualData = rawData[rawData.shape[0]-horizon:, 4]
rawData = rawData[:rawData.shape[0]-horizon, :]

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,+1))
rawData[:, 4] = minMax.fit_transform(rawData[:, 4])

# Available data - to store known and unknown
availableData = np.zeros((rawData.shape[0]+horizon,4))
availableData[:rawData.shape[0], 0:3] = rawData[:, 0:3]
availableData[:rawData.shape[0], 3] = rawData[:, valueIndex]

#Pre-Process the data to form feature and target vectors
tsp = ts.TimeSeriesProcessor(rawData[:, valueIndex], depth, 1)
processedData = tsp.getProcessedData()

#Append bias
processedData = np.hstack((np.ones((processedData.shape[0], 1)),processedData))

#Training data
trainingInputData = processedData[:,:1+depth]
trainingOutputData = processedData[:,1+depth:]


#Validation data
validationInputData = trainingInputData
validationOutputData = trainingOutputData


#Tune the reservoir
spectralRadiusBound = (0.0, 1.0)
inputScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 300
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

# To predict the test data, predict training and valiadation data as a warm up
trainingPredictedOutputData = res.predict(trainingInputData)


#Now, start predicting the future
xAxis = []
predictedOutput = []
lastDayIndex = rawData.shape[0] -1
lastDay = date(int(availableData[lastDayIndex, 0]), int(availableData[lastDayIndex, 1]), int(availableData[lastDayIndex, 2]))
for i in range(horizon):
    nextDay = lastDay + timedelta(days=1)
    nextDayIndex = lastDayIndex + 1
    year = nextDay.strftime("%Y")
    month = nextDay.strftime("%m")
    day = nextDay.strftime("%d")
    nextDayStr = "Date.UTC(" + year+","+ str((int(month)-1)) + "," + day +")"
    xAxis.append(nextDayStr)

    #Compose the query
    query = [1.0]
    for d in range(1,depth+1):
        query.append(availableData[nextDayIndex-d,3])

    nextDayPredicted = res.predict(np.array(query).reshape((1, depth+1)))
    predictedOutput.append(nextDayPredicted[0,0])

    #Update the available data
    availableData[nextDayIndex,3] = nextDayPredicted[0,0]
    availableData[nextDayIndex,0] = int(year)
    availableData[nextDayIndex,1] = int(month)
    availableData[nextDayIndex,2] = int(day)

    lastDay = nextDay
    lastDayIndex = nextDayIndex

predicted = minMax.inverse_transform(np.array(predictedOutput))

# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxis), actualData)
outplot.setSeries('Predicted Output', np.array(xAxis), predicted)
outplot.createOutput()