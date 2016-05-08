from utility import Utility
from datetime import datetime
import sys
from performance import ErrorMetrics as metrics
import pandas as pd
from reservoir import Utility as utilRes, classicESN as ESN, ReservoirTopology as topology, ActivationFunctions as act
import os
from plotting import OutputPlot as outputPlot
import numpy as np
from sklearn import preprocessing as pp
from utility import EnhancedUtility as eutil

# Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "taylor_swift"
datasetFileName = directoryName + profileName + ".csv"


# Forecasting parameters
depth = 1
horizon = 7

util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to daily)
resampledSeries = util.resampleSeriesSum(series, "D")
del series

# Remove the outliers
resampledSeries = util.detectAndRemoveOutliers(resampledSeries)

# Step 3 - Rescale the series
normalizedSeries = util.scaleSeriesStandard(resampledSeries)
del resampledSeries

# Step 4 - Split the data into training and testing series
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries, horizon)

# Step 4 - Form feature and target vectors
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectors(trainingSeries, depth)

# Input-to-reservoir fully connected
size = 300
initialTransient = 0
inputWeight = topology.ClassicInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size).generateWeightMatrix()

# Reservoir-to-reservoir fully connected
reservoirWeight = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()

# Tune the reservoir
tuner = eutil.ReservoirParameterTuner(size=size,
                                      initialTransient=initialTransient,
                                      trainingInputData=featureVectors,
                                      trainingOutputData=targetVectors,
                                      initialSeedSeries=trainingSeries,
                                      validationOutputData=testingSeries.values,
                                      arbitraryDepth=depth,
                                      spectralRadiusBound=(0.0,1.25),
                                      inputScalingBound=(0.0,1.0),
                                      reservoirScalingBound=(0.0,1.0),
                                      leakingRateBound=(0.0,1.0),
                                      inputWeightMatrix=inputWeight,
                                      reservoirWeightMatrix=reservoirWeight)

spectralRadiusOptimum, inputScalingOptimium, reservoirScalingOptimum, leakingRateOptimum = tuner.getOptimalParameters()


res = ESN.Reservoir(size=size,
                    inputData=featureVectors,
                    outputData=targetVectors,
                    spectralRadius=spectralRadiusOptimum,
                    inputScaling=inputScalingOptimium,
                    reservoirScaling=reservoirScalingOptimum,
                    leakingRate=leakingRateOptimum,
                    initialTransient=0,
                    inputWeightRandom=inputWeight,
                    reservoirWeightRandom=reservoirWeight)
res.trainReservoir()

# Warm up
predictedTrainingOutputData = res.predict(featureVectors[:initialTransient])

# Predict the future
predictedTestingOutputData = util.predictFutureDays(res, trainingSeries, depth, horizon)

error = metrics.RootMeanSquareError().compute(testingSeries.values.flatten().reshape(horizon,1), predictedTestingOutputData.values.flatten().reshape(horizon,1))
print("Regression error:"+str(error))

# Step 8 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedTestingOutputData)

# Step 9 - Plot the results
details = profileName + "_horizon_" + str(horizon) + "_depth_" + str(depth) + "_network_size_" + str(size)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")




