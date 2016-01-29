#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import ClassicESN as reservoir, ClassicTuner as tuner
from plotting import OutputPlot as outputPlot
from performance import RootMeanSquareError as rmse
from sklearn import preprocessing as pp
import numpy as np
from reservoir import Utility as util

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data)

#Get only 4000 points
data = data[:5000].reshape((5000, 1))

# Split the data into training, validation and testing
trainingData, validationData, testingData = util.splitData(data, 0.4, 0.4, 0.2)
nValidation = validationData.shape[0]
nTesting = testingData.shape[0]

# Form feature vectors for training data
trainingInputData, trainingOutputData = util.formFeatureVectors(trainingData)
validationInputData, validationOutputData = util.formFeatureVectors(validationData)

spectralRadiusBound = (0.0, 1.00)
inputScalingBound = (0.0, 1.0)
reservoirScalingBound = (0.0, 1.0)
leakingRateBound = (0.0, 1.0)
size = 256
initialTransient = 50
resTuner = tuner.ReservoirTuner(size=size,
                                initialTransient=initialTransient,
                                trainingInputData=trainingInputData,
                                trainingOutputData=trainingOutputData,
                                validationInputData=validationInputData,
                                validationOutputData=validationOutputData,
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
                          outputData=trainingOutputData,
                          inputWeightRandom=inputWeightOptimum,
                          reservoirWeightRandom=reservoirWeightOptimum)
res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(validationInputData)

# Predict
predictedTestOutputData = util.predictFuture(res, validationData[-1], nTesting)

#Plotting of the prediction output and error
outplot = outputPlot.OutputPlot("Outputs/Prediction.html", "Mackey-Glass Time Series", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', testingData[:nTesting, 0])
outplot.setYSeries('Predicted Output', predictedTestOutputData[:nTesting, 0])
outplot.createOutput()
print("Done!")