from utility import Utility
from datetime import datetime

# Dataset
directoryName = "Datasets/"
profileName = "Jeep"
datasetFileName = directoryName + profileName + "_time_interaction_rate.csv"

daysOfHorizon = 14
daysOfDepth = 30
horizon = 24*daysOfHorizon#14 days ahead
depth = 24*daysOfDepth #30 days
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
util.trainESNWithCorrelationTuned(size=networkParameters['size'], featureVectors=bestFeatures, targetVectors=targetVectors,
                           initialTransient=networkParameters['initialTransient'],
                           inputConnectivity=networkParameters['inputConnectivity'], reservoirConnectivity=networkParameters['reservoirConnectivity'],
                           inputScaling=networkParameters['inputScaling'], reservoirScaling=networkParameters['reservoirScaling'],
                           spectralRadius=networkParameters['spectralRadius'], leakingRate=networkParameters['leakingRate'])


# Step 8 - Predict the future
predictedSeries = util.predict(util.esn, availableSeries, networkParameters['arbitraryDepth'], horizon, bestFeaturesIndices)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)


# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) +  "_network_size_" + str(networkParameters['size'])
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
