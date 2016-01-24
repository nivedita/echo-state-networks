#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import EchoStateNetwork as ESN, Tuner as tuner
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np
from reservoir import ReservoirTopology as topology

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

#Training Input data - get 2000 points
nTraining = 2000
nTesting = 1000

trainingInputData = data[:nTraining].reshape((nTraining,1))
trainingOutputData = data[1:nTraining+1].reshape((nTraining,1))

# Partition the training data into training and validation
trainingRatio = 0.8
splitIndex = int(trainingInputData.shape[0] * trainingRatio)
trainingInputData = trainingInputData[:splitIndex]
validationInputData = trainingInputData[splitIndex:]
trainingOutputData = trainingOutputData[:splitIndex]
validationOutputData = trainingOutputData[splitIndex:]

testingInputData = data[nTraining:nTraining+nTesting]
testActualOutputData = data[nTraining+1:nTraining+nTesting+1]



trainingInputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
trainingOutputData = data[1:nTraining+1].reshape((nTraining, 1))

# Partition the training data into training and validation
trainingRatio = 0.8
splitIndex = int(trainingInputData.shape[0] * trainingRatio)
trainingInputData = trainingInputData[:splitIndex]
validationInputData = trainingInputData[splitIndex:]
trainingOutputData = trainingOutputData[:splitIndex]
validationOutputData = trainingOutputData[splitIndex:]

#Testing Input and output data
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))

# Tune the Erdoys Renyi Network
size = 100
initialTransient = 5
inputConnectivityBound = (0.1,0.9) # Usually dense
probabilityBound = (0.0,1.0)
errorFunction = rmse.RootMeanSquareError()
esnTuner = tuner.ESNErdosTuner(size=size,
                            initialTransient=initialTransient,
                            trainingInputData=trainingInputData,
                            trainingOutputData=trainingOutputData,
                            validationInputData=validationInputData,
                            validationOutputData=validationOutputData,
                            inputConnectivityBound=inputConnectivityBound,
                            probabilityBound=probabilityBound,
                            times=5)
inputConnectivityOptimum, probabilityOptimum = esnTuner.getOptimalParameters()

res = ESN.EchoStateNetwork(size=size,
                           inputData=trainingInputData,
                           outputData=trainingOutputData,
                           reservoirTopology=topology.ErdosRenyiTopology(size=size, probability=probabilityOptimum),
                           inputConnectivity=inputConnectivityOptimum)
res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(trainingInputData)


#Predict future values
predictedTestOutputData = []
lastAvailableData = testInputData[0, 1]
for i in range(nTesting):
    query = [1.0]
    query.append(lastAvailableData)

    #Predict the next point
    nextPoint = res.predict(np.array(query).reshape((1,2)))[0,0]
    predictedTestOutputData.append(nextPoint)

    lastAvailableData = nextPoint

predictedTestOutputData = np.array(predictedTestOutputData).reshape((nTesting, 1))

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Mackey-Glass Time Series", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testActualOutputData[:nTesting, 0])
outplot.setYSeries('Predicted Output', predictedTestOutputData[:nTesting, 0])
outplot.createOutput()
print("Done!")