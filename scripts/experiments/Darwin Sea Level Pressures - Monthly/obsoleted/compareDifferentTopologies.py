
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import DetermimisticReservoir as dr, Tuner as tuner, EchoStateNetwork as ESN, DeterministicTuner as dTuner, ReservoirTopology as topology
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import ErrorMetrics as rmse
import numpy as np
from sklearn import preprocessing as pp
import os
from datetime import datetime

errorFunction = rmse.MeanSquareError()

#Read data from the file
data = np.loadtxt('darwin.slp.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((0,1))
data = minMax.fit_transform(data)

#Input data and output data
nTraining = 1150
nTesting = 200

# Training input data
inputTrainingData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputTrainingData = data[1:nTraining+1].reshape((nTraining, 1))


# Partition the training data into training and validation
trainingRatio = 0.6
splitIndex = int(inputTrainingData.shape[0] * trainingRatio)
trainingInputData = inputTrainingData[:splitIndex]
validationInputData = inputTrainingData[splitIndex:]
trainingOutputData = outputTrainingData[:splitIndex]
validationOutputData = outputTrainingData[splitIndex:]

# trainingInputData = inputTrainingData
# trainingOutputData = outputTrainingData
# validationInputData = inputTrainingData
# validationOutputData = outputTrainingData

# Testing Input data
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))

size = 200
initialTransient = 5
runTimes = 10
inputConnectivity = 0.6

def runStandardESN():
    standardError = 0
    testPredictedOutputDataStandard = 0
    for i in range(runTimes):
        #Tune the standard reservoir
        reservoirConnectivityBound = (0.1,1.0)

        resTuner = tuner.ESNConnTuner(size=size,
                                     initialTransient=initialTransient,
                                     trainingInputData=trainingInputData,
                                     trainingOutputData=trainingOutputData,
                                     validationInputData=validationInputData,
                                     validationOutputData=validationOutputData,
                                     inputConnectivity=inputConnectivity,
                                     reservoirConnectivityBound=reservoirConnectivityBound,
                                     times=10)

        reservoirConnectivityOptimum = resTuner.getOptimalParameters()


        #Train the reservoir with optimal parameters
        esn = ESN.EchoStateNetwork(size=size,
                                   inputData=trainingInputData,
                                   outputData=trainingOutputData,
                                   reservoirTopology=topology.RandomTopology(size=size, connectivity=reservoirConnectivityOptimum),
                                   inputConnectivity=inputConnectivity)



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
        standardError += errorFunction.compute(actual.reshape((actual.shape[0],1)), testPredictedOutputDataStandard.reshape((testPredictedOutputDataStandard.shape[0],1)))
    return testPredictedOutputDataStandard, (standardError/runTimes)

def runErdosESN():
    erdosError = 0
    testPredictedOutputDataErdos = 0
    for i in range(runTimes):
        # Tune the Erdoys Renyi Network
        probabilityBound = (0.1,1.0) #To avoid isolated bounds, keep the upper bound low
        esnTuner = tuner.ESNErdosTuner(size=size,
                                    initialTransient=initialTransient,
                                    trainingInputData=trainingInputData,
                                    trainingOutputData=trainingOutputData,
                                    validationInputData=validationInputData,
                                    validationOutputData=validationOutputData,
                                    inputConnectivity=inputConnectivity,
                                    probabilityBound=probabilityBound,
                                    times=10)
        probabilityOptimum = esnTuner.getOptimalParameters()

        res = ESN.EchoStateNetwork(size=size,
                                   inputData=trainingInputData,
                                   outputData=trainingOutputData,
                                   reservoirTopology=topology.ErdosRenyiTopology(size=size, probability=probabilityOptimum),
                                   inputConnectivity=inputConnectivity)
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
        testPredictedOutputDataErdos = minMax.inverse_transform(testingPredictedOutputData)

        #Error
        erdosError += errorFunction.compute(actual.reshape((actual.shape[0], 1)), testPredictedOutputDataErdos.reshape((testPredictedOutputDataErdos.shape[0],1)))
    return testPredictedOutputDataErdos, (erdosError/runTimes)

def runSmallWorld():
    smallWorldErrorError = 0
    testPredictedOutputDataSmallWorld = 0
    for i in range(runTimes):
        # Tune the Small world graphs
        meanDegreeBound = (2, size-1)
        betaBound = (0.1, 1.0)
        esnTuner = tuner.ESNSmallWorldGraphsTuner(size=size,
                                                  initialTransient=initialTransient,
                                                  trainingInputData=trainingInputData,
                                                  trainingOutputData=trainingOutputData,
                                                  validationInputData=validationInputData,
                                                  validationOutputData=validationOutputData,
                                                  inputConnectivity=inputConnectivity,
                                                  meanDegreeBound=meanDegreeBound,
                                                  betaBound=betaBound,
                                                  times=10)

        meanDegreeOptimum, betaOptimum = esnTuner.getOptimalParameters()

        res = ESN.EchoStateNetwork(size=size,
                                   inputData=trainingInputData,
                                   outputData=trainingOutputData,
                                   reservoirTopology=topology.SmallWorldGraphs(size=size, meanDegree=meanDegreeOptimum, beta=betaOptimum),
                                   inputConnectivity=inputConnectivity)
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
        testPredictedOutputDataSmallWorld = minMax.inverse_transform(testingPredictedOutputData)

        #Error
        smallWorldErrorError += errorFunction.compute(actual.reshape((actual.shape[0], 1)), testPredictedOutputDataSmallWorld.reshape((testPredictedOutputDataSmallWorld.shape[0],1)))
    return testPredictedOutputDataSmallWorld, (smallWorldErrorError/runTimes)

def runScaleFree():
    scaleFreeError = 0
    testPredictedOutputDataScaleFree = 0
    for i in range(runTimes):
        # Tune the scale free networks
        attachmentBound = (1,size-1)
        esnTuner = tuner.ESNScaleFreeNetworksTuner(size=size,
                                                  initialTransient=initialTransient,
                                                  trainingInputData=trainingInputData,
                                                  trainingOutputData=trainingOutputData,
                                                  validationInputData=validationInputData,
                                                  validationOutputData=validationOutputData,
                                                  inputConnectivity=inputConnectivity,
                                                  attachmentBound=attachmentBound)

        attachmentOptimum = esnTuner.getOptimalParameters()

        res = ESN.EchoStateNetwork(size=size,
                                   inputData=trainingInputData,
                                   outputData=trainingOutputData,
                                   reservoirTopology=topology.ScaleFreeNetworks(size=size, attachmentCount=attachmentOptimum),
                                   inputConnectivity=inputConnectivity)
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
        testPredictedOutputDataScaleFree = minMax.inverse_transform(testingPredictedOutputData)

        #Error
        scaleFreeError += errorFunction.compute(actual.reshape((actual.shape[0], 1)), testPredictedOutputDataScaleFree.reshape((testPredictedOutputDataScaleFree.shape[0],1)))
    return testPredictedOutputDataScaleFree, (scaleFreeError/runTimes)

testPredictedOutputDataStandard, standardError = runStandardESN()
testPredictedOutputDataErdos, erdosError = runErdosESN()
testPredictedOutputDataSmallWorld, smallWorldError = runSmallWorld()
testPredictedOutputDataScaleFree, scaleFreeError = runScaleFree()
testActualOutputData = minMax.inverse_transform(testActualOutputData[:nTesting, 0])

outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Darwin Sea Level Pressure Prediction", "Comparison of Random Graph Topolgies - Standard Vs Erdos", "Time", "Sea Level Pressure")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testActualOutputData)
outplot.setYSeries('Predicted Output_standard_ESN_with_all_parameters_tuned', testPredictedOutputDataStandard)
outplot.setYSeries('Predicted Output_Erdoys_ESN_with_parameters_tuned', testPredictedOutputDataErdos)
outplot.setYSeries('Predicted Output_Small World_ESN_with_parameters_tuned', testPredictedOutputDataSmallWorld)
outplot.setYSeries('Predicted Output_Scale_Free_with_parameters_tuned', testPredictedOutputDataScaleFree)
outplot.createOutput()

#Plotting of regression error
errPlot = errorPlot.ErrorPlot(outputFolderName + "/RegressionError.html", "Comparison of different graph topologies", "with parameters tuner", "ESN Configuration", "Total Error")
errPlot.setXAxis(np.array(['Standard', 'Erdos Renyi', 'Small World Graph', 'Scale Free Network']))
errPlot.setYAxis('RMSE', np.array([standardError, erdosError, smallWorldError, scaleFreeError]))
errPlot.createOutput()
