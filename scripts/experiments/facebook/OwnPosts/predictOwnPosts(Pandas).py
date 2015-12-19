import numpy as np
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
from sklearn import preprocessing as pp
from reservoir import Reservoir as reservoir, Tuner as tuner
from datetime import datetime
from plotting import OutputTimeSeries as outTimePlot
import os

# Read the data
df = pd.read_csv('facebookPostsCount_bmw_time.csv', index_col=0, parse_dates=True)

horizon = 90

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
featureIntervalList = [pd.Timedelta(days=-1), pd.Timedelta(days=-2), pd.Timedelta(days=-3), pd.Timedelta(days=-4), pd.Timedelta(days=-5), pd.Timedelta(days=-6), pd.Timedelta(days=-7),
                       pd.Timedelta(weeks=-1), pd.Timedelta(weeks=-2), pd.Timedelta(weeks=-3), pd.Timedelta(weeks=-4)]

depth =

targetIntervalList = [pd.Timedelta(days=0)]

# Pre-process the data and form feature and target vectors
tsp = tsi.TimeSeriesIntervalProcessor(series, featureIntervalList, targetIntervalList)
featureVectors, targetVectors = tsp.getProcessedData()

#Append bias to feature vectors
featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)),featureVectors))

print(featureVectors)
print(targetVectors)

# #Tune the reservoir
# spectralRadiusBound = (0.0, 1.0)
# inputScalingBound = (0.0, 1.0)
# reservoirScalingBound = (0.0, 1.0)
# leakingRateBound = (0.0, 1.0)
# size = 300
# initialTransient = 50
# resTuner = tuner.ReservoirTuner(size=size,
#                                   initialTransient=initialTransient,
#                                   trainingInputData=featureVectors,
#                                   trainingOutputData=targetVectors,
#                                   validationInputData=featureVectors,
#                                   validationOutputData=targetVectors,
#                                   spectralRadiusBound=spectralRadiusBound,
#                                   inputScalingBound=inputScalingBound,
#                                   reservoirScalingBound=reservoirScalingBound,
#                                   leakingRateBound=leakingRateBound)
# spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightOptimum, reservoirWeightOptimum = resTuner.getOptimalParameters()

#Train the reservoir with the optimal parameters
res = reservoir.Reservoir(size=300,
                         spectralRadius=0.5,
                         inputScaling=0.5,
                         reservoirScaling=0.5,
                         leakingRate=0.3,
                         initialTransient=50,
                         inputData=featureVectors,
                         outputData=targetVectors)

res.trainReservoir()

#Predict for the training data as a warmup
trainingPredictedOutputData = res.predict(featureVectors)

#Now, start predicted the future

startDate = series.last_valid_index()
predicted = []
xAxis = []
for i in range(horizon):
    nextDate = startDate + pd.Timedelta(days=1)
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

    predictedValue = res.predict(feature)[0,0]
    predicted.append(predictedValue)

    #Add it to the series
    series[nextDate] = predictedValue


actualData = minMax.inverse_transform(testingData)
predicted = minMax.inverse_transform(np.array(predicted))

# Plotting of the actual and prediction output
outputFolderName = "Outputs/Outputs-Pandas" + str(datetime.now()) + "_horizon_" + str(horizon)
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Facebook Own posts-BMW", "", "Number of posts")
outplot.setSeries('Actual Output', np.array(xAxis), testingData)
outplot.setSeries('Predicted Output', np.array(xAxis), predicted)
outplot.createOutput()




