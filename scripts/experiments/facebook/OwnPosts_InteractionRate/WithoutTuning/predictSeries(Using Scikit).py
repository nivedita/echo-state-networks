from sklearn.svm import SVR
from utility import Utility
from datetime import datetime
import sys

#Get the commamd line arguments
directoryName = "Datasets/"
profileName = sys.argv[1]
datasetFileName = directoryName + profileName + "_time_interaction_rate.csv"

daysOfHorizon = 30
daysOfDepth = 14
horizon = 24*daysOfHorizon#7 days ahead
depth = 24*daysOfDepth #30 days
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to hourly)
resampledSeries = util.resampleSeriesMean(series, "H")
del series



# Todo - Train based on the recent data only.
yearsOfData = 5
recentCount = yearsOfData * 365 * 24 + horizon
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries


# Step 3 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries

# Step 4 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

# Step 5 - Form the feature and target vectors for training
featureTrainingVectors, targetTrainingVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, depth)

