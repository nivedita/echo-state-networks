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

#Read data from the file
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')

data = rawData[:rawData.shape[0], rawData.shape[1] -1].reshape((rawData.shape[0], 1))

nTraining = 2100
nTesting = data.shape[0] - nTraining -1

# inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining, :data.shape[1] -1].reshape((nTraining, data.shape[1] - 1))))
# outputData = data[:nTraining, data.shape[1] -1].reshape((nTraining, 1))

inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputData = data[1:nTraining+1].reshape((nTraining, 1))

#Train
inputWeightRandom = np.load("Outputs/inputWeight.npy")
reservoirWeightRandom = np.load("Outputs/reservoirWeight.npy")
res = reservoir.Reservoir(size = 600, spectralRadius = 1.10, inputScaling = 0.55, leakingRate = 0.30, initialTransient = 0, inputData = inputData, outputData = outputData, inputWeightRandom=inputWeightRandom, reservoirWeightRandom=reservoirWeightRandom)
# res.inputWeightRandom = np.load("Outputs/inputWeight.npy")
# res.reservoirWeightRandom = np.load("Outputs/reservoirWeight.npy")
res.trainReservoir()

inputWeightRandom = np.load("Outputs/inputWeight.npy")
reservoirWeightRandom = np.load("Outputs/reservoirWeight.npy")

# testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting, :data.shape[1] -1].reshape((nTesting, data.shape[1] - 1))))
# testActualOutputData = data[nTraining:nTraining+nTesting, data.shape[1] -1].reshape((nTesting, 1))

testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))

#Predict
testPredictedOutputData = res.predict(testInputData)

#Form the date for x series
xData = rawData[nTraining:nTraining+nTesting, :3]
xAxis = []
xAxisNew = []
for i in range(xData.shape[0]):
    year = str(int(xData[i, 0]))
    month = str(int(xData[i, 1]))
    day = str(int(xData[i, 2]))
    xAxis.append("'" + year +"-"+ month + "-" + day +"'")
    xAxisNew.append("Date.UTC(" + year +","+ month + "," + day +")")


outputFolderName = "Outputs" + str(datetime.now())
os.mkdir(outputFolderName)

# #Plotting of the prediction output and error
# outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Day", "Likes Count")
# outplot.setXSeries(np.array(xAxis))
# outplot.setYSeries('Actual Output', testActualOutputData[:nTesting, 0])
# outplot.setYSeries('Predicted Output', testPredictedOutputData[:nTesting, 0])
# outplot.createOutput()


#Plotting of the prediction output and error
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/Prediction.html", "Likes count for facebook page-BMW", "", "Likes Count")
outplot.setSeries('Actual Output', np.array(xAxisNew), testActualOutputData[:nTesting, 0])
outplot.setSeries('Predicted Output', np.array(xAxisNew), testPredictedOutputData[:nTesting, 0])
outplot.createOutput()

#Save the input weight and reservoir weight
np.save(outputFolderName + "/inputWeight", res.inputWeightRandom)
np.save(outputFolderName + "/reservoirWeight", res.reservoirWeightRandom)