from utility import Utility
from datetime import datetime
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np

# Network Parameters
networkParameters = {}
networkParameters['initialTransient'] = 50
networkParameters['spectralRadius'] = 0.79
networkParameters['inputScaling'] = 0.5
networkParameters['reservoirScaling'] = 0.5
networkParameters['leakingRate'] = 0.3
networkParameters['inputConnectivity'] = 1.0
networkParameters['reservoirConnectivity'] = 0.1
networkParameters['arbitraryDepth'] = 24 * 60 # Depth of 1 year

# Dataset
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

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

networkParameters['size'] = int(trainingSeries.shape[0]/10) # One-tenth of length of the training set
#networkParameters['size'] = 256

# Step 6 - Form the feature and target vectors for training
bestDepth, bestFeaturesIndices, bestFeatures, targetVectors = util.getBestFeatures(trainingSeries, validationSeries, networkParameters, Utility.FeatureSelectionMethod.Pattern_Analysis, args={'threshold': 0.5})


# Step 7 - Tune the leaking rate by brute force
optimalLeakingRate = util.getBestLeakingRate(bestFeaturesIndices, bestDepth, bestFeatures, targetVectors, trainingSeries,
                                             validationSeries, networkParameters)

# Step 7 - Tune and Train the network
util.trainESNWithoutTuning(size=networkParameters['size'], featureVectors=bestFeatures, targetVectors=targetVectors, initialTransient=networkParameters['initialTransient'],
                           inputConnectivity=networkParameters['inputConnectivity'], reservoirConnectivity=networkParameters['reservoirConnectivity'],
                           inputScaling=networkParameters['inputScaling'], reservoirScaling=networkParameters['reservoirScaling'],
                           spectralRadius=networkParameters['spectralRadius'], leakingRate=networkParameters['leakingRate'])


# Step 8 - Predict the future
predictedSeries = util.predict(util.esn, availableSeries, bestDepth, horizon, bestFeaturesIndices)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)


# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) +  "_network_size_" + str(networkParameters['size'])
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
