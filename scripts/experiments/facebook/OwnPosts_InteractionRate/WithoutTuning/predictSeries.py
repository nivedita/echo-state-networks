from utility import Utility
from datetime import datetime
import sys
from performance import ErrorMetrics as metrics


#Get the commamd line arguments
directoryName = "Datasets/"
profileName = sys.argv[1]
datasetFileName = directoryName + profileName + "_time_interaction_rate.csv"

daysOfHorizon = 10
daysOfDepth = 7
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


# # Step 3 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries
#normalizedSeries = filteredSeries

# Step 4 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

# Step 5 - Form the feature and target vectors for training
featureTrainingVectors, targetTrainingVectors = util.formContinousFeatureAndTargetVectors(trainingSeries, depth)


# Step 6 - Train the network
networkSize = 256
util.trainESNWithoutTuning(size=networkSize, featureVectors=featureTrainingVectors, targetVectors=targetTrainingVectors,
                            initialTransient=50, inputConnectivity=1.0, reservoirConnectivity=0.5,
                            inputScaling=0.0, reservoirScaling=0.0, spectralRadius=0.79, leakingRate=0.45)

# util.trainESNWithMinimalTuning(size=1500,
#                                featureTrainingVectors=featureTrainingVectors,
#                                targetTrainingVectors=targetTrainingVectors,
#                                featureValidationVectors=featureValidationVectors,
#                                targetValidationVectors=targetValidationVectors,
#                                initialTransient=50)

# Step 7 - Predict the future
predictedSeries = util.predictFuture(trainingSeries, depth, horizon)


# # Step 8 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)
#actualSeries = testingSeries

error = metrics.MeanSquareError()
regressionError = error.compute(testingSeries.values, predictedSeries.values)
print(regressionError)


# Step 9 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth) + "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
