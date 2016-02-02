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
horizon = 30
valueIndex = 4

# Read data from the file
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')

# Read until the horizon
actualData = rawData[rawData.shape[0]-horizon:, valueIndex]
rawData = rawData[:rawData.shape[0]-horizon, :]

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,+1))
rawData[:, 4] = minMax.fit_transform(rawData[:, 4])

#Just vary the depths between 100 and 300 in steps of 50
depthList = range(50,450,50)
predictedDict = {} #Dict of predicted arrays. Depth being the key
regressionError = []#List of regression errors
xAxisError = []
for depth in depthList:
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


    deScaled = minMax.inverse_transform(np.array(predictedOutput))
    predictedDict[str(depth)] = deScaled

    #Calculate the error
    errorFunction = rmse.RootMeanSquareError()
    error = errorFunction.compute(actualData.reshape(1, horizon), deScaled.reshape(1,horizon))
    regressionError.append(error)


    xAxisError.append("Depth_"+str(depth))

# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now()) + "_depth_comparison_"+"_horizon_" + str(horizon)
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Comparison of prediction outputs", "with varying depths", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxis), actualData)
for depth in depthList:
    seriesName = 'Predicted_Output_depth' + str(depth)
    predicted = predictedDict[str(depth)]
    outplot.setSeries(seriesName, np.array(xAxis), predicted)
outplot.createOutput()

#Plotting of regression
errPlot = errorPlot.ErrorPlot(outputFolderName +"/Regression_Error.html", "Comparison of regression error", "with varying depths", "ESN Configuration", "Total Error")
#X-axis
errPlot.setXAxis(np.array(xAxisError))
#Series data
errPlot.setYAxis('Total Regression Error', np.array(regressionError))
errPlot.createOutput()
