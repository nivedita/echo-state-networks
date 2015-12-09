#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir, Tuner as tuner
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

#Training Input data - get 2000 points
nTraining = 2000
nTesting = 2000
trainingInputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))
trainingOutputData = data[1:nTraining+1].reshape((nTraining, 1))

#Testing Input and output data
testInputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTesting, 1))

initialTransient = 100
spectralRadiusBound = (0.0, 1.25)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 256
initialTransient = 50
resTuner = tuner.ReservoirTuner(size=size,
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

#Train
res = reservoir.Reservoir(size=size,
                          spectralRadius=spectralRadiusOptimum,
                          inputScaling=inputScalingOptimum,
                          reservoirScaling=reservoirScalingOptimum,
                          leakingRate=leakingRateOptimum,
                          initialTransient=initialTransient,
                          inputData=trainingInputData,
                          outputData=trainingOutputData)
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