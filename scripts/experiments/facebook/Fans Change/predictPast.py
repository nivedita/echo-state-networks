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

#Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "taylor_swift"
datasetFileName = directoryName + profileName + ".csv"


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

# Step 4 - Form feature and target vectors
depth = 1
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsInOrder(normalizedSeries, depth)

n = featureVectors.shape[0]


# Input-to-reservoir fully connected
size = 100
inputWeight = topology.ClassicInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size).generateWeightMatrix()

# Reservoir-to-reservoir fully connected
#reservoirWeight = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()
reservoirWeight = topology.SmallWorldGraphs(size=size, meanDegree=int(size/2), beta=0.8).generateWeightMatrix()

res = ESN.Reservoir(size=size,
                    inputData=featureVectors,
                    outputData=targetVectors,
                    spectralRadius=1.5,
                    inputScaling=0.1,
                    reservoirScaling=0.5,
                    leakingRate=0.7,
                    initialTransient=0,
                    inputWeightRandom=inputWeight,
                    reservoirWeightRandom=reservoirWeight)
res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(featureVectors)

predictedSeries = pd.Series(data=predictedTrainingOutputData.flatten(), index=normalizedSeries.index[-predictedTrainingOutputData.shape[0]:])

predictedSeries = util.descaleSeries(predictedSeries)
actualSeries = util.descaleSeries(normalizedSeries)

# Step 9 - Plot the results
details = profileName + "_depth_" + str(depth) + "_network_size_" + str(size)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Series", "Predicted Series"], "Facebook Fans Change - "+profileName, "Fans Change")







