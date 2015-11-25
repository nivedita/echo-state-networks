
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np

#Read data from the file
data = np.loadtxt('darwin.slp.txt')

#Input data - get 2000 points
nTraining = 1000
inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputData = data[1:nTraining+1].reshape((nTraining, 1))

#Train, Predict and Measure
reservoirObj = []
predictedOutput = []
error = []
nTesting = 300
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))
errorFunction = rmse.RootMeanSquareError()
sizes = range(10, 50, 10)


for i in sizes:
    #Train
    res = reservoir.Reservoir(size = i, spectralRadius = 0.95, inputScaling = 0.5, leakingRate = 0.3, initialTransient = 5, inputData = inputData, outputData = outputData)
    res.trainReservoir()
    reservoirObj.append(res)

    #Predict
    prediction = res.predict(testInputData)
    predictedOutput.append(prediction)

    #Measure performance
    error.append(errorFunction.compute(testActualOutputData, prediction))


#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Darwin Sea Level Pressure Prediction", "Comparison of different echo state networks", "Time", "Sea Level Pressure")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testActualOutputData[:nTesting, 0])
errplot = errorPlot.ErrorPlot("Outputs/RegressionError.html", "Darwin Sea Level Pressure Prediction", "Comparison of different echo state networks", "ESN Configuration", "Root Mean Square Error")

xAxis = []
for i in range(len(reservoirObj)):
    seriesName = "Predicted Output for ESN with " + str(reservoirObj[i].Nx) +" nodes"
    seriesData = predictedOutput[i][:nTesting, 0]
    outplot.setYSeries(seriesName, seriesData)

    xAxis.append("With " +str(reservoirObj[i].Nx) + " nodes")
outplot.createOutput()
errplot.setXAxis(np.array(xAxis))
errplot.setYAxis('Total Regression Error', np.array(error))
errplot.createOutput()




