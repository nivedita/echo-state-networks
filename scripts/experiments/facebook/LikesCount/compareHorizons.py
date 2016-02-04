#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import EchoStateNetwork as reservoir, Tuner as tuner
from plotting import OutputTimeSeries as outTimePlot, ErrorPlot as errorPlot
from datetime import date, timedelta, datetime
import numpy as np
import os
from timeseries import TimeSeries as ts
from sklearn import preprocessing as pp
from performance import ErrorMetrics as rmse

# Depth and Horizon
depth = 100
valueIndex = 4
maxHorizon = 100

# Read data from the file
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')

# Read until the horizon
actualData = rawData[rawData.shape[0]-maxHorizon:, valueIndex]
data = rawData[:rawData.shape[0]-maxHorizon, :]

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,+1))
data[:, 4] = minMax.fit_transform(data[:, 4])


#Pre-Process the data to form feature and target vectors
tsp = ts.TimeSeriesProcessor(data[:, valueIndex], depth, 1)
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
reservoirScalingBound = (0.0, 1.0)
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
                                reservoirScalingBound=reservoirScalingBound,
                                leakingRateBound=leakingRateBound)
spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightOptimum, reservoirWeightOptimum = resTuner.getOptimalParameters()

#Just vary the horizons between 10 and 100 in steps of 10
horizonList = range(10,maxHorizon+10,10)
xAxisDict = {}
predictedDict = {} #Dict of predicted arrays. Depth being the key
regressionError = []#List of regression errors
xAxisError = []
for horizon in horizonList:
    #Train the reservoir with the optimal parameters
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

    # To predict the test data, predict training and validation data as a warm up - TODO - Is it really needed?
    trainingPredictedOutputData = res.predict(trainingInputData)

    # Available data - to store known and unknown
    availableData = np.zeros((data.shape[0]+horizon,4))
    availableData[:data.shape[0], 0:3] = data[:, 0:3]
    availableData[:data.shape[0], 3] = data[:, valueIndex]

    #Actual data
    actualDataLim = actualData[0:horizon]

    #Now, start predicting the future
    xAxis = []
    predictedOutput = []
    lastDayIndex = data.shape[0] -1
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


    deScaled = minMax.inverse_transform(np.array(predictedOutput))
    predictedDict[str(horizon)] = deScaled
    xAxisDict[str(horizon)] = xAxis

    #Calculate the error
    errorFunction = rmse.MeanSquareError()
    error = errorFunction.compute(actualDataLim.reshape(1, horizon), deScaled.reshape(1,horizon))
    regressionError.append(error)


    xAxisError.append("Horizon_"+str(horizon))

# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now()) + "_horizon_comparison_"+"_depth_" + str(100)
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Comparison of prediction outputs", "with varying horizons", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxis), actualData)
for horizon in horizonList:
    seriesName = 'Predicted_Output_horizon' + str(horizon)
    predicted = predictedDict[str(horizon)]
    xAxis = xAxisDict[str(horizon)]
    outplot.setSeries(seriesName, np.array(xAxis), predicted)
outplot.createOutput()

#Plotting of regression
errPlot = errorPlot.ErrorPlot(outputFolderName +"/Regression_Error.html", "Comparison of regression error", "with varying horizons", "ESN Configuration", "Total Error")
#X-axis
errPlot.setXAxis(np.array(xAxisError))
#Series data
errPlot.setYAxis('Total Regression Error', np.array(regressionError))
errPlot.createOutput()
