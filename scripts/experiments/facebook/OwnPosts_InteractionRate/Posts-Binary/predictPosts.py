from utility import Utility
from datetime import datetime
import sys
from performance import ErrorMetrics as metrics
from reservoir import ActivationFunctions as act
import numpy as np


#Get the commamd line arguments
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

daysOfHorizon = 14
daysOfDepth = 60
horizon = 24*daysOfHorizon#7 days ahead
depth = 24*daysOfDepth #30 days
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)


# Step 2 - Resample the series (to hourly)
resampledSeries = util.resampleSeriesExists(series, "H")
del series

# Todo - Train based on the recent data only.
yearsOfData = 3
recentCount = yearsOfData * 365 * 24 + horizon
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries


# # Step 3 - Scale the series
#normalizedSeries = util.scaleSeries(filteredSeries)
normalizedSeries = filteredSeries
del filteredSeries

# Step 4 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

# Step 5 - Form the feature and target vectors for training
featureIndices = []
numberOfDays = 60
interval = 24
arbitraryDepth = numberOfDays * interval
for i in range(numberOfDays - 1):
    featureIndices.append(interval*i)
featureIndices = np.array(featureIndices)

featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, arbitraryDepth)
featureVectors = featureVectors[:, featureIndices]
featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)),featureVectors))

# Step 6 - Train the network
networkSize = int(featureVectors.shape[0]/10)
util.trainESNWithoutTuning(size=networkSize, featureVectors=featureVectors, targetVectors=targetVectors,
                           initialTransient=50, inputConnectivity=1.0, reservoirConnectivity=0.3,
                           inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3,
                           reservoirActivationFunction=act.HyperbolicTangent(),
                           outputActivationFunction=act.ReLU())

# Step 7 - Predict the future
predictedSeries = util.predict(util.esn, trainingSeries, arbitraryDepth, horizon, featureIndices)


# # Step 8 - De-scale the series
# actualSeries = util.descaleSeries(testingSeries)
# predictedSeries = util.descaleSeries(predictedSeries)

actualSeries = testingSeries


error = metrics.MeanSquareError()
regressionError = error.compute(testingSeries.values, predictedSeries.values)
print(regressionError)


# Step 9 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth) + "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
