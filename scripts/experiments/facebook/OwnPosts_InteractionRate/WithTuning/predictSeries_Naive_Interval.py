from utility import Utility
from datetime import datetime
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np

# Network Parameters
networkParameters = {}
networkParameters['size'] = 2000
networkParameters['initialTransient'] = 50
networkParameters['spectralRadius'] = 0.79
networkParameters['inputScaling'] = 0.5
networkParameters['reservoirScaling'] = 0.5
networkParameters['leakingRate'] = 0.3
networkParameters['reservoirConnectivity'] = 0.3
networkParameters['arbitraryDepth'] = 24 * 90 # Depth of 1 year

# Dataset
directoryName = "Datasets/"
profileName = "Dodge"
datasetFileName = directoryName + profileName + "_time_interaction_rate.csv"

# Horizon - used to split the training and testing
daysOfHorizon = 14 # 10 days ahead
horizon = 24*daysOfHorizon
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Re-sample the series (to hourly)
resampledSeries = util.resampleSeriesSum(series, "H")
del series

# Step 3 - Filter recent data
yearsOfData = 3
recentCount = yearsOfData * 365 * 24 + horizon
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries

# Step 4 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries

# Step 5 - Split into training, validation and testing series
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries, horizon)
availableSeries = trainingSeries
trainingSeries, validationSeries = util.splitIntoTrainingAndValidationSeries(trainingSeries, trainingSetRatio=0.9)

# Step 6 - Form the feature and target vectors for training
bestFeaturesIndices, bestFeatures, targetVectors = util.getBestFeatures(trainingSeries, validationSeries, networkParameters)


# Step 7 - Train the network
networkSize = 2000
util.trainESNWithoutTuning(size=networkParameters['size'], featureVectors=bestFeatures, targetVectors=targetVectors,
                           initialTransient=networkParameters['initialTransient'], reservoirConnectivity=networkParameters['reservoirConnectivity'],
                           inputScaling=networkParameters['inputScaling'], reservoirScaling=networkParameters['reservoirScaling'],
                           spectralRadius=networkParameters['spectralRadius'], leakingRate=networkParameters['leakingRate'])


# Step 8 - Predict the future
predictedSeries = util.predict(util.esn, availableSeries, networkParameters['arbitraryDepth'], horizon, bestFeaturesIndices)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)


# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) +  "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
