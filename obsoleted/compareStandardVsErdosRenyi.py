
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import DetermimisticReservoir as dr, Tuner as tuner, EchoStateNetwork as esn, DeterministicTuner as dTuner, ReservoirTopology as topology
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import ErrorMetrics as rmse
import numpy as np
from sklearn import preprocessing as pp

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((0,1))
data = minMax.fit_transform(data)

#Training Input data - get 2000 points
nTraining = 2000
nTesting = 2000
inputTrainingData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputTrainingData = data[1:nTraining+1].reshape((nTraining, 1))

#Testing Input and output data
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))


size = 100
initialTransient = 5

#Run for different deterministic topologies
topologyNames = ['Erdos Renyi - 0.1', 'Erdos Renyi - 0.3', 'Erdos Renyi - 0.5', 'Erdos Renyi - 0.7', 'Erdos Renyi - 0.9']
topologyObjects = [topology.ErdosRenyiTopology(size=size, probability=0.1),topology.ErdosRenyiTopology(size=size, probability=0.3),
                   topology.ErdosRenyiTopology(size=size, probability=0.5), topology.ErdosRenyiTopology(size=size, probability=0.7),
                   topology.ErdosRenyiTopology(size=size, probability=0.9)
]
topologyTestOutput = []
topologyError = []
errorFunction = rmse.RootMeanSquareError()

for i in range(len(topologyObjects)):

    res = esn.EchoStateNetwork(size=size,
                               inputData=inputTrainingData,
                               outputData=outputTrainingData,
                               reservoirTopology=topologyObjects[i],
                               inputConnectivity=0.8)
    res.trainReservoir()

    #Warm up using training data
    trainingPredictedOutputData = res.predict(inputTrainingData)

    #Predict for future
    lastAvailablePoint = testInputData[0,1]
    testingPredictedOutputData = []
    for i in range(nTesting):
        #Compose the query
        query = [1.0]
        query.append(lastAvailablePoint)

        #Predict the next point
        nextPoint = res.predict(np.array(query).reshape(1,2))[0,0]
        testingPredictedOutputData.append(nextPoint)

        lastAvailablePoint = nextPoint

    testingPredictedOutputData = np.array(testingPredictedOutputData).reshape(nTesting, 1)

    #De-normalize
    actual = minMax.inverse_transform(testActualOutputData)
    testPredictedOutputData = minMax.inverse_transform(testingPredictedOutputData)
    topologyTestOutput.append(testPredictedOutputData)

    #Error
    topologyError.append(errorFunction.compute(actual.reshape((actual.shape[0], 1)), testPredictedOutputData.reshape((testPredictedOutputData.shape[0],1))))


#Tune the standard reservoir
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 100
initialTransient = 5
randomTopology = topology.RandomTopology(size=size, connectivity=0.6)
inputConnectivity = 0.8
inputConnectivityBound = (0.1, 0.9)
reservoirConnectivityBound = (0.1,0.9)

resTuner = tuner.ESNTunerWithConnectivity(size=size,
                          initialTransient=initialTransient,
                          trainingInputData=inputTrainingData,
                          trainingOutputData=outputTrainingData,
                          validationInputData=inputTrainingData,
                          validationOutputData=outputTrainingData,
                          spectralRadiusBound=spectralRadiusBound,
                          inputScalingBound=inputScalingBound,
                          reservoirScalingBound=reservoirScalingBound,
                          leakingRateBound=leakingRateBound,
                          inputConnectivityBound=inputConnectivityBound,
                          reservoirConnectivityBound=reservoirConnectivityBound)

spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightConn, reservoirConn = resTuner.getOptimalParameters()


#Train the reservoir with optimal parameters
esn = esn.EchoStateNetwork(size=size,
                           inputData=inputTrainingData,
                           outputData=outputTrainingData,
                           reservoirTopology=topology,
                           spectralRadius=spectralRadiusOptimum,
                           inputScaling=inputScalingOptimum,
                           reservoirScaling=reservoirScalingOptimum,
                           leakingRate=leakingRateOptimum,
                           initialTransient=initialTransient,
                           inputConnectivity=inputConnectivity,
                           inputWeightConn=inputWeightConn,
                           reservoirWeightConn=reservoirConn)


esn.trainReservoir()

#Warm up for the trained data
predictedTrainingOutputData = esn.predict(inputTrainingData)


#Predict for future
lastAvailablePoint = predictedTrainingOutputData[nTraining-1,0]
testingPredictedOutputData = []
for i in range(nTesting):
    #Compose the query
    query = [1.0]
    query.append(lastAvailablePoint)

    #Predict the next point
    nextPoint = esn.predict(np.array(query).reshape(1,2))[0,0]
    testingPredictedOutputData.append(nextPoint)

    lastAvailablePoint = nextPoint

testingPredictedOutputData = np.array(testingPredictedOutputData).reshape(nTesting, 1)

#Predict
testPredictedOutputDataStandard = minMax.inverse_transform(testingPredictedOutputData)
actual = minMax.inverse_transform(testActualOutputData)
standardError = errorFunction.compute(actual.reshape((actual.shape[0],1)), testPredictedOutputDataStandard.reshape((testPredictedOutputDataStandard.shape[0],1)))
testActualOutputData = minMax.inverse_transform(testActualOutputData[:nTesting, 0])

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Mackey-Glass Prediction", "Comparison of Random Graph Topolgies - Standard Vs Erdos", "Time", "Sea Level Pressure")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testActualOutputData)
outplot.setYSeries('Predicted Output_standard_ESN_with_parameters_tuned', testPredictedOutputDataStandard)
for i in range(len(topologyObjects)):
    seriesName = 'Predicted Output_'+ topologyNames[i]
    seriesData = topologyTestOutput[i]
    outplot.setYSeries(seriesName, seriesData)
outplot.createOutput()

#Plotting of regression error
topologyNames.append('Standard')
topologyError.append(standardError)
errPlot = errorPlot.ErrorPlot("Outputs/RegressionError.html", "Comparison of standard vs deterministic", "with different parameters", "ESN Configuration", "Total Error")
errPlot.setXAxis(np.array(topologyNames))
errPlot.setYAxis('RMSE', np.array(topologyError))
errPlot.createOutput()