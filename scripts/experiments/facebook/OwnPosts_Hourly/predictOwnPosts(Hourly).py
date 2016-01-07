import pandas as pd
import numpy as np
from plotting import OutputTimeSeries as plotting
import os
from datetime import datetime
from reservoir import EchoStateNetwork as esn, Tuner as tuner, ReservoirTopology as topology
from sklearn import preprocessing as pp
from timeseries import TimeSeriesInterval as tsi

def _sum(x):
    if len(x) == 0:
        return 0
    else:
        return sum(x)

#Parameters
horizon = 24*4 #4 days ahead
depth = 24*30 #30 days
data = 'facebookPosts_timestamp_bmw_time.csv'

# Read the data
df = pd.read_csv(data, index_col=0, parse_dates=True, names=['value'])

# Convert the dataframe into series
data = np.array(df.as_matrix()).flatten().astype(float)
series = pd.Series(data=data, index=df.index)
resampled_series = series.resample(rule='H', how=_sum)

# Normalize the data
data = resampled_series.values
minMax = pp.MinMaxScaler((0,1))
normalizedData = minMax.fit_transform(data)


# Split the data into training and testing data
index = resampled_series.index
trainingData = normalizedData[:normalizedData.shape[0]-horizon]
trainingIndex = index[:normalizedData.shape[0]-horizon]
testingData = normalizedData[normalizedData.shape[0]-horizon:]

# Form the timeseries data
trainingSeries = pd.Series(data=trainingData.flatten(),index=trainingIndex)

# Feature list
featureIntervalList = []
for i in range(depth, 0, -1):
    interval = pd.Timedelta(hours=-(i))
    featureIntervalList.append(interval)

targetIntervalList = [pd.Timedelta(hours=0)]

# Pre-process the data and form feature and target vectors
tsp = tsi.TimeSeriesIntervalProcessor(trainingSeries, featureIntervalList, targetIntervalList)
featureVectors, targetVectors = tsp.getProcessedData()

#Append bias to feature vectors
featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)),featureVectors))

#Train the reservoir with the optimal parameters
size = 1500
initialTransient = 50
network = esn.EchoStateNetwork(size=size,
                               inputData=featureVectors,
                               outputData=targetVectors,
                               reservoirTopology=topology.RandomTopology(size=size, connectivity=0.4),
                               spectralRadius=0.79,
                               inputConnectivity=0.8)
network.trainReservoir()

# #Tune and Train
# spectralRadiusBound = (0.0, 1.0)
# inputScalingBound = (0.0, 1.0)
# reservoirScalingBound = (0.0, 1.0)
# leakingRateBound = (0.0, 1.0)
# inputConnectivityBound = (0.1,1.0) #Usually, densely connected
# reservoirConnectivityBound = (0.1,0.9) #Usually, sparsely connected
# size = 1000
# initialTransient = 5
# inputConnectivity = 0.7
# esnTuner = tuner.ESNConnTuner(size=size,
#                      initialTransient=initialTransient,
#                      trainingInputData=featureVectors,
#                      trainingOutputData=targetVectors,
#                      validationInputData=featureVectors,
#                      validationOutputData=targetVectors,
#                      inputConnectivityBound=inputConnectivityBound,
#                      reservoirConnectivityBound=reservoirConnectivityBound)
# inputConnectivityOptimum, reservoirConnectivityOptimum = esnTuner.getOptimalParameters()
#
# reservoirTopology = topology.RandomTopology(size=size, connectivity=reservoirConnectivityOptimum)
#
# network = esn.EchoStateNetwork(size=size,
#                                inputData=featureVectors,
#                                outputData=targetVectors,
#                                reservoirTopology=reservoirTopology,
#                                inputConnectivity=inputConnectivityOptimum)
# network.trainReservoir()

#Predict for the training data as a warmup
trainingPredictedOutputData = network.predict(featureVectors)

#Now, start predicted the future

nextValue = trainingSeries.last_valid_index()
predicted = []
xAxis = []
for i in range(horizon):
    nextValue = nextValue + pd.Timedelta(hours=1)
    year = nextValue.strftime("%Y")
    month = nextValue.strftime("%m")
    day = nextValue.strftime("%d")
    hour = nextValue.strftime("%H")
    nextDayStr = "Date.UTC(" + str(year)+","+ str(int(month)-1) + "," + str(day) + ","+ str(hour)+")"
    xAxis.append(nextDayStr)

    #Form the feature vectors
    feature = [1.0]
    for interval in featureIntervalList:
        feature.append(trainingSeries[nextValue + interval])

    feature = np.array(feature).reshape((1,len(featureIntervalList)+1))

    predictedValue = network.predict(feature)[0,0]
    predicted.append(predictedValue)

    #Add it to the series
    trainingSeries[nextValue] = predictedValue


actualData = minMax.inverse_transform(testingData.reshape((testingData.shape[0],1)))[:,0]
predicted = minMax.inverse_transform(np.array(predicted))

# Plotting of the actual and prediction output
outputFolderName = "Outputs/Outputs-Pandas_weekly_daily" + str(datetime.now()) + "_horizon_" + str(horizon)
os.mkdir(outputFolderName)
outplot = plotting.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Facebook Own posts-BMW", "", "Number of posts")
outplot.setSeries('Actual Output', np.array(xAxis), actualData)
outplot.setSeries('Predicted Output', np.array(xAxis), predicted)
outplot.createOutput()






