from utility import Utility
from datetime import datetime
import sys


#Get the commamd line arguments
directoryName = "Datasets/"
profileName = sys.argv[1]
datasetFileName = directoryName + profileName + "_time_interaction_rate.csv"

daysOfHorizon = 10
daysOfDepth = 14
horizon = 24*daysOfHorizon#7 days ahead
depth = 24*daysOfDepth #30 days
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to hourly)
resampledSeries = util.resampleSeriesSum(series, "H")
del series

# Todo - Train based on the recent data only.
yearsOfData = 5
recentCount = yearsOfData * 365 * 24 + horizon
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries


# Step 3 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries
#normalizedSeries = filteredSeries

# Step 4 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

# Step 5 - Split the training into training and validation
trainingSeries, validationSeries = util.splitIntoTrainingAndValidationSeries(trainingSeries, 0.95)

# Step 6 - Form the feature and target vectors for training
featureTrainingVectors, targetTrainingVectors = util.formContinousFeatureAndTargetVectors(trainingSeries, depth)

# Step 7 - Tune and train the network
networkSize = 256
initialTransient = 50
initialSeedSeries = trainingSeries[-depth:]
validationOutputData = validationSeries.values
util.trainESNWithTuning(size=networkSize, featureVectors=featureTrainingVectors, targetVectors=targetTrainingVectors,
                        initialTransient=initialTransient, initialSeedSeries=initialSeedSeries, depth=depth,
                        validationOutputData=validationOutputData)

# Step 7 - Predict the future
predictedSeries = util.predictFuture(trainingSeries, depth, horizon)

# Step 8 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)
#actualSeries = testingSeries

# Step 9 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth) + "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
