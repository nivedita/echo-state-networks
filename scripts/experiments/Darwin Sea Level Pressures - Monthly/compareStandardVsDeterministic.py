
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import DetermimisticReservoir as dr, Tuner as tuner, Reservoir as reservoir, DeterministicTuner as dTuner
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np
from sklearn import preprocessing as pp
from performance import DataWhittener as dw


#Read data from the file
data = np.loadtxt('darwin.slp.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((0,1))
data = minMax.fit_transform(data)

#Input data and output data
nTraining = 1000
nTesting = 100
inputTrainingData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputTrainingData = data[1:nTraining+1].reshape((nTraining, 1))

testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))
size = 100
initialTransient = 5

#Run for different deterministic topologies
topologyNames = ['DLR', 'DLRB', 'SCR']
topologyObjects = [dr.ReservoirTopologyDLR(0.7), dr.ReservoirTopologyDLRB(0.7, 0.3), dr.ReservoirTopologySCR(0.3)]
topologyTestOutput = []
topologyError = []
errorFunction = rmse.RootMeanSquareError()

for i in range(len(topologyObjects)):

    #Tune the parameters
    inputWeight = 0.1
    leakingRateBound = (0.0,1.0)
    inputScalingBound = (0.0,1.0)
    resTuner = dTuner.DeterministicReservoirTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=inputTrainingData,
                                                 trainingOutputData=outputTrainingData,
                                                 validationInputData=inputTrainingData,
                                                 validationOutputData=outputTrainingData,
                                                 inputWeight_v=inputWeight,
                                                 reservoirTopology=topologyObjects[i],
                                                 inputScalingBound=inputScalingBound,
                                                 leakingRateBound=leakingRateBound)

    inputScalingOptimum, leakingRateOptimum = resTuner.getOptimalParameters()

    #Train the reservoir with the optimal parameters
    res = dr.DeterministicReservoir(size=size,
                                    inputWeight_v=0.1,
                                    inputWeightScaling=0.2,
                                    inputData=inputTrainingData,
                                    outputData=outputTrainingData,
                                    leakingRate=0.2,
                                    initialTransient=5,
                                    reservoirTopology=topologyObjects[i])
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
    testPredictedOutputData = minMax.inverse_transform(testingPredictedOutputData)
    topologyTestOutput.append(testPredictedOutputData)

    #Error
    topologyError.append(errorFunction.compute(testActualOutputData.reshape((testActualOutputData.shape[0], 1)), testPredictedOutputData.reshape((testPredictedOutputData.shape[0],1))))


#Tune the standard reservoir
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 100
initialTransient = 5
resTuner = tuner.ReservoirTuner(size = size,
                                initialTransient=initialTransient,
                                trainingInputData=inputTrainingData,
                                trainingOutputData=outputTrainingData,
                                validationInputData=inputTrainingData,
                                validationOutputData=outputTrainingData,
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
                          inputData=inputTrainingData,
                          outputData=outputTrainingData,
                          inputWeightRandom=inputWeightOptimum,
                          reservoirWeightRandom=reservoirWeightOptimum)
res.trainReservoir()

#Warm up for the trained data
predictedTrainingOutputData = res.predict(inputTrainingData)


#Predict for future
lastAvailablePoint = predictedTrainingOutputData[nTraining-1,0]
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

#Predict
testPredictedOutputDataStandard = minMax.inverse_transform(testingPredictedOutputData)
standardError = errorFunction.compute(testActualOutputData.reshape((testActualOutputData.shape[0],1)), testPredictedOutputDataStandard.reshape((testPredictedOutputDataStandard.shape[0],1)))
testActualOutputData = minMax.inverse_transform(testActualOutputData[:nTesting, 0])

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Darwin Sea Level Pressure Prediction", "Comparison of standard vs deterministic", "Time", "Sea Level Pressure")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testActualOutputData)
outplot.setYSeries('Predicted Output_standard_ESN', testPredictedOutputDataStandard)
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
