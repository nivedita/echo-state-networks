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
depth = 500
horizon = 1
valueIndex = 4
tsp = ts.TimeSeriesProcessor(rawData, depth, horizon, valueIndex)
processedData = tsp.getProcessedData()

inputData = np.hstack((np.ones((processedData.shape[0], 1)),processedData[:processedData.shape[0],:depth]))
outputData = processedData[:processedData.shape[0],depth:depth+horizon]

# Train
inputWeightRandom = np.load("Outputs/inputWeight.npy")
reservoirWeightRandom = np.load("Outputs/reservoirWeight.npy")
res = reservoir.Reservoir(size=100, spectralRadius=0.5, inputScaling=0.10, leakingRate=0.3, initialTransient=10, inputData=inputData, outputData=outputData)
res.trainReservoir()

#Predict for trained data - just to see how it fits
xAxisPast = []
testPredictedOutputDataPast = []
testActualOutputDataPast = []

index = depth
while index <= (rawData.shape[0] - horizon):
    #Day for the prediction
    day = "Date.UTC(" + str(int(rawData[index, 0])) +","+ str(int(rawData[index, 1]) -1) + "," + str(int(rawData[index, 2])) +")"

    #Actual value
    actual = rawData[index, valueIndex]

    #Predicted value
    #Compose the query
    query = [1.0]
    depthValues = range(depth, 0, -1)
    for d in depthValues:
        query.append(rawData[index - d, valueIndex])

    pred = res.predict(np.array(query).reshape((1, 1+depth)))

    #Add to the list
    xAxisPast.append(day)
    testActualOutputDataPast.append(actual)
    testPredictedOutputDataPast.append(pred[0, 0])

    index += 1


# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxisPast), np.array(testActualOutputDataPast))
outplot.setSeries('Predicted Output', np.array(xAxisPast), np.array(testPredictedOutputDataPast))
outplot.createOutput()

# Save the input weight and reservoir weight
np.save(outputFolderName + "/inputWeight", res.inputWeightRandom)
np.save(outputFolderName + "/reservoirWeight", res.reservoirWeightRandom)