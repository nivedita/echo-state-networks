
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir, Tuner as tuner
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np
from sklearn import preprocessing as pp

#Read data from the file
data = np.loadtxt('darwin.slp.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((0,1))
data = minMax.fit_transform(data)

#Training data - get 2000 points
nTraining = 1000
trainingInputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
trainingOutputData = data[1:nTraining+1].reshape((nTraining, 1))


#Testing data
nTesting = 300
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))


#Tune and Train
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 100
initialTransient = 5
resTuner = tuner.ReservoirTuner(size = size,
                                initialTransient=initialTransient,
                                trainingInputData=trainingInputData,
                                trainingOutputData=trainingOutputData,
                                validationInputData=trainingInputData,
                                validationOutputData=trainingOutputData,
                                spectralRadiusBound=spectralRadiusBound,
                                inputScalingBound=inputScalingBound,
                                reservoirScalingBound=reservoirScalingBound,
                                leakingRateBound=leakingRateBound)
spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightOptimum, reservoirWeightOptimum = resTuner.getOptimalParameters()

res = reservoir.Reservoir(size=size,
                          spectralRadius=spectralRadiusOptimum,
                          inputScaling=inputScalingOptimum,
                          reservoirScaling=reservoirScalingOptimum,
                          leakingRate=leakingRateOptimum,
                          initialTransient=initialTransient,
                          inputData=trainingInputData,
                          outputData=trainingOutputData)
res.trainReservoir()

#Warm up for the trained data
predictedTrainingOutputData = res.predict(trainingInputData)


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
#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Darwin Sea Level Pressure Prediction", "Prediction of future values", "Time", "Sea Level Pressure")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(testActualOutputData[:nTesting, 0]))
outplot.setYSeries('Predicted Output', minMax.inverse_transform(testingPredictedOutputData[:nTesting, 0]))
outplot.createOutput()



