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

# Set the cut off point for training
# For instance, let's test the horizon points
# So, split last horizon points at the end
depth = 300
horizon = 30
cutOffIndex = rawData.shape[0] - horizon
dataForTraining = rawData[:cutOffIndex, :]
dataForTesting = rawData[cutOffIndex:, :]

valueIndex = 4
tsp = ts.TimeSeriesProcessor(dataForTraining, depth, horizon, valueIndex)
processedData = tsp.getProcessedData()

inputWeightRandom = np.load("Outputs/inputWeight.npy")
reservoirWeightRandom = np.load("Outputs/reservoirWeight.npy")
inputData = np.hstack((np.ones((processedData.shape[0], 1)),processedData[:processedData.shape[0],:depth]))
outputData = processedData[:processedData.shape[0],depth:depth+horizon]

# Train
res = reservoir.Reservoir(size=100, spectralRadius=1.00, inputScaling=0.10, leakingRate=0.3, initialTransient=50, inputData=inputData, outputData=outputData)
res.trainReservoir()

#Predict for future
#Compose the query
query = [1.0]
depthValues = range(depth, 0, -1)
for d in depthValues:
    query.append(rawData[cutOffIndex - d, valueIndex])

testPredictedOutputData = res.predict(np.array(query).reshape((1, 1+depth)))

#Create the needed lists for plotting
xAxis = []
testActualOutputData = rawData[cutOffIndex:, valueIndex]

index = cutOffIndex
while index < rawData.shape[0]:
    day = "Date.UTC(" + str(int(rawData[index, 0])) +","+ str(int(rawData[index, 1]) -1) + "," + str(int(rawData[index, 2])) +")"
    xAxis.append(day)
    index += 1

# Plotting of the actual and prediction output
outputFolderName = "Outputs" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxis), testActualOutputData)
outplot.setSeries('Predicted Output', np.array(xAxis), testPredictedOutputData.reshape((horizon, 1))[ :,0])
outplot.createOutput()

# Save the input weight and reservoir weight
np.save(outputFolderName + "/inputWeight", res.inputWeightRandom)
np.save(outputFolderName + "/reservoirWeight", res.reservoirWeightRandom)