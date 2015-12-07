
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import DetermimisticReservoir as dr, Tuner as tuner, Reservoir as reservoir
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

# #Whiten the data
# whitener = dw.DataWhittener()
# data = whitener.WhittenData(data.reshape((data.shape[0],1)))

#Input data and output data
nTraining = 1000
nTesting = 300
inputTrainingData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
outputTrainingData = data[1:nTraining+1].reshape((nTraining, 1))

testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))
size = 1000
initialTransient = 5

#Run for different deterministic topologies
topologyNames = ['DLR', 'DLRB', 'SCR']
topologyObjects = [dr.ReservoirTopologyDLR(0.7), dr.ReservoirTopologyDLRB(0.7, 0.3), dr.ReservoirTopologySCR(0.7)]
topologyTestOutput = []
topologyError = []
errorFunction = rmse.RootMeanSquareError()

for i in range(len(topologyObjects)):
    #Train the reservoir with the optimal parameters
    res = dr.DeterministicReservoir(size=size,
                                    inputWeight_v=0.2,
                                    inputWeightScaling=0.5,
                                    inputData=inputTrainingData,
                                    outputData=outputTrainingData,
                                    leakingRate=0.5,
                                    initialTransient=5,
                                    reservoirTopology=topologyObjects[i])
    res.trainReservoir()

    #Warm up
    trainingPredictedOutputData = res.predict(inputTrainingData)

    #Predict
    testPredictedOutputData = res.predict(testInputData)

    #De-normalize
    testPredictedOutputData = minMax.inverse_transform(testPredictedOutputData[:nTesting, 0])
    topologyTestOutput.append(testPredictedOutputData)

    #Error
    topologyError.append(errorFunction.compute(testActualOutputData.reshape((testActualOutputData.shape[0], 1)), testPredictedOutputData.reshape((testPredictedOutputData.shape[0],1))))

    # #De-whiten
    # testActualOutputData = whitener.UnwhittenData(testActualOutputData[:nTesting, 0].reshape((testActualOutputData.shape[0],1)))
    # testPredictedOutputData = whitener.UnwhittenData(testPredictedOutputData[:nTesting, 0].reshape((testPredictedOutputData.shape[0],1)))


#Tune the standard reservoir
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
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
                                leakingRateBound=leakingRateBound)
spectralRadiusOptimum, inputScalingOptimum, leakingRateOptimum, inputWeightOptimum, reservoirWeightOptimum = resTuner.getOptimalParameters()

#Train the reservoir with the optimal parameters
res = reservoir.Reservoir(size=size,
                          spectralRadius=spectralRadiusOptimum,
                          inputScaling=inputScalingOptimum,
                          leakingRate=leakingRateOptimum,
                          initialTransient=initialTransient,
                          inputData=inputTrainingData,
                          outputData=outputTrainingData,
                          inputWeightRandom=inputWeightOptimum,
                          reservoirWeightRandom=reservoirWeightOptimum)
res.trainReservoir()

#Warm up
trainingPredictedOutputData = res.predict(inputTrainingData)[:nTesting, 0]

#Predict
testPredictedOutputDataStandard = minMax.inverse_transform(res.predict(testInputData))
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
