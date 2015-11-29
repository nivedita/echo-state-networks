#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot, OutputTimeSeries as outTimePlot
from performance import RootMeanSquareError as rmse
from datetime import datetime
import numpy as np
import os
from datetime import date, timedelta
from timeseries import TimeSeries as ts

# Read data from the file
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')

data = rawData[:rawData.shape[0], rawData.shape[1] -1].reshape((rawData.shape[0], 1))

depth = 1
horizon = 1
tsp = ts.TimeSeriesProcessor(rawData, depth, horizon, 4)
processedData = tsp.getProcessedData()

inputData = np.hstack((np.ones((processedData.shape[0], 1)),processedData[:processedData.shape[0],:depth]))
outputData = processedData[:processedData.shape[0],depth:depth+horizon]

# Train
inputWeightRandom = np.load("Outputs/inputWeight.npy")
reservoirWeightRandom = np.load("Outputs/reservoirWeight.npy")
res = reservoir.Reservoir(size=600, spectralRadius=1.25, inputScaling=0.50, leakingRate=0.20, initialTransient=0, inputData=inputData, outputData=outputData, inputWeightRandom = inputWeightRandom, reservoirWeightRandom = reservoirWeightRandom)
res.trainReservoir()

#Predict for past
lastDayIndex = 0
lastDayValue = rawData[lastDayIndex, 4]

xAxisPast = []
testPredictedOutputDataPast = []
testActualOutputDataPast = []
numberOfDaysInPast = rawData.shape[0]-1
for i in range(numberOfDaysInPast):
    nextDayIndex = lastDayIndex + 1
    nextDay = "Date.UTC(" + str(int(rawData[nextDayIndex, 0])) +","+ str(int(rawData[nextDayIndex, 1]) -1) + "," + str(int(rawData[nextDayIndex, 2])) +")"
    nextDayValue = rawData[nextDayIndex, 4]
    nextDayPred = res.predict(np.array([1.0, lastDayValue]).reshape((1, 2)))

    # Add it to the list
    xAxisPast.append(nextDay)
    testActualOutputDataPast.append(nextDayValue)
    testPredictedOutputDataPast.append(nextDayPred[0,0])

    lastDayIndex = nextDayIndex
    lastDayValue = nextDayValue


#Predict for future - for 90 days in advance
lastIndex = rawData.shape[0] - 1
lastDay = date(int(rawData[lastIndex, 0]), int(rawData[lastIndex, 1]), int(rawData[lastIndex, 2]))
lastDayValue = rawData[lastIndex, 4]

xAxisFuture = []
testPredictedOutputDataFuture = []
numberOfDaysInFuture = 90
for i in range(numberOfDaysInFuture):
    nextDay = lastDay + timedelta(days=1)
    nextDayPred = res.predict(np.array([1.0, lastDayValue]).reshape((1, 2)))

    # Add it to the list
    xAxisFuture.append("Date.UTC(" + str(nextDay.year) +","+ str(nextDay.month -1) + "," + str(nextDay.day) +")")
    testPredictedOutputDataFuture.append(nextDayPred[0, 0])

    lastDay = nextDay
    lastDayValue = nextDayPred[0, 0]

# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxisPast), np.array(testActualOutputDataPast))
outplot.setSeries('Predicted Output', np.array(xAxisPast+xAxisFuture), np.array(testPredictedOutputDataPast+testPredictedOutputDataFuture))
outplot.createOutput()

# Save the input weight and reservoir weight
np.save(outputFolderName + "/inputWeight", res.inputWeightRandom)
np.save(outputFolderName + "/reservoirWeight", res.reservoirWeightRandom)