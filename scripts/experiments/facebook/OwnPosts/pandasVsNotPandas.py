#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import EchoStateNetwork as reservoir, Tuner as tuner
from plotting import OutputTimeSeries as outTimePlot
from datetime import date, timedelta, datetime
import numpy as np
import os
from timeseries import TimeSeries as ts
from sklearn import preprocessing as pp

# Depth and Horizon
depth = 100
horizon = 90
valueIndex = 3

# Read data from the file
rawData = np.loadtxt("facebookPostsCount_bmw_raw.txt", delimiter=',')

# Normalize the raw data
minMax = pp.MinMaxScaler((0,1))
rawData[:, valueIndex] = minMax.fit_transform(rawData[:, valueIndex])

# Read until the horizon
actualData = rawData[rawData.shape[0]-horizon:, valueIndex]
rawData = rawData[:rawData.shape[0]-horizon, :]

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
spectralRadius = 0.5
inputScaling = 0.5
reservoirScaling = 0.5
leakingRate = 0.3
size = 500
initialTransient = 50

#Train the reservoir with the optimal parameters
res1 = reservoir.Reservoir(size=size,
                         spectralRadius=spectralRadius,
                         inputScaling=inputScaling,
                         reservoirScaling=reservoirScaling,
                         leakingRate=leakingRate,
                         initialTransient=initialTransient,
                         inputData=trainingInputData,
                         outputData=trainingOutputData)

res1.trainReservoir()

# To predict the test data, predict training and valiadation data as a warm up
trainingPredictedOutputData = res1.predict(trainingInputData)


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

    nextDayPredicted = res1.predict(np.array(query).reshape((1, depth+1)))
    predictedOutput.append(nextDayPredicted[0,0])

    #Update the available data
    availableData[nextDayIndex,3] = nextDayPredicted[0,0]
    availableData[nextDayIndex,0] = int(year)
    availableData[nextDayIndex,1] = int(month)
    availableData[nextDayIndex,2] = int(day)

    lastDay = nextDay
    lastDayIndex = nextDayIndex

predicted1 = minMax.inverse_transform(np.array(predictedOutput))
actualData = minMax.inverse_transform(actualData)



import pandas as pd
from timeseries import TimeSeriesInterval as tsi
#For Pandas
# Read the data
df = pd.read_csv('facebookPostsCount_bmw_time.csv', index_col=0, parse_dates=True)

# Normalize the data
data = np.array(df.as_matrix())
data = data.reshape((data.shape[0],1)).astype(float)
minMax = pp.MinMaxScaler((0,1))
data = minMax.fit_transform(data)

#Split the data for training and testing
trainingData = data[:data.shape[0]-horizon, :]
trainingIndex = df.index[:data.shape[0]-horizon]
testingData = data[data.shape[0]-horizon:, :]

# Form the timeseries data
series = pd.Series(data=trainingData.flatten(),index=trainingIndex)

# Feature and Target interval lists
# featureIntervalList = [pd.Timedelta(days=-1), pd.Timedelta(days=-2), pd.Timedelta(days=-3), pd.Timedelta(days=-4), pd.Timedelta(days=-5), pd.Timedelta(days=-6), pd.Timedelta(days=-7),
#                        pd.Timedelta(weeks=-1), pd.Timedelta(weeks=-2), pd.Timedelta(weeks=-3), pd.Timedelta(weeks=-4)]

featureIntervalList = []
for i in range(depth, 0, -1):
    interval = pd.Timedelta(days=-(i))
    featureIntervalList.append(interval)

targetIntervalList = [pd.Timedelta(days=0)]

# Pre-process the data and form feature and target vectors
tsp = tsi.TimeSeriesIntervalProcessor(series, featureIntervalList, targetIntervalList)
featureVectors, targetVectors = tsp.getProcessedData()

#Append bias to feature vectors
featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)),featureVectors))

print(featureVectors)
print(targetVectors)

#Train the reservoir with the optimal parameters
res2 = reservoir.Reservoir(size=size,
                         spectralRadius=spectralRadius,
                         inputScaling=inputScaling,
                         reservoirScaling=reservoirScaling,
                         leakingRate=leakingRate,
                         initialTransient=initialTransient,
                         inputData=featureVectors,
                         outputData=targetVectors,
                         inputWeightRandom=res1.inputWeightRandom,
                         reservoirWeightRandom=res1.reservoirWeightRandom)

res2.trainReservoir()

#Predict for the training data as a warmup
trainingPredictedOutputData = res2.predict(featureVectors)

#Now, start predicted the future

nextDate = series.last_valid_index()
predicted = []
xAxis = []
for i in range(horizon):
    nextDate = nextDate + pd.Timedelta(days=1)
    year = nextDate.strftime("%Y")
    month = nextDate.strftime("%m")
    day = nextDate.strftime("%d")
    nextDayStr = "Date.UTC(" + year+","+ str((int(month)-1)) + "," + day +")"
    xAxis.append(nextDayStr)

    #Form the feature vectors
    feature = [1.0]
    for interval in featureIntervalList:
        feature.append(series[nextDate + interval])

    feature = np.array(feature).reshape((1,len(featureIntervalList)+1))

    predictedValue = res2.predict(feature)[0,0]
    predicted.append(predictedValue)

    #Add it to the series
    series[nextDate] = predictedValue


predicted2 = minMax.inverse_transform(np.array(predicted))

# Plotting of the actual and prediction output
outputFolderName = "Outputs/Outputs" + str(datetime.now()) + "_depth_" + str(depth) + "_horizon_" + str(horizon)
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Facebook Own posts-BMW", "", "Number of posts")
outplot.setSeries('Actual Output', np.array(xAxis), actualData)
outplot.setSeries('Predicted Output 1', np.array(xAxis), predicted1)
outplot.setSeries('Predicted Output 2', np.array(xAxis), predicted2)
outplot.createOutput()

