from utility import Utility
from datetime import datetime


datasetFileName = "facebookPosts_audi_time_photo_interaction.csv"
daysOfHorizon = 4
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
recentCount = yearsOfData * 365 * 24 + horizon #1 year of data for training+ horizon number of test data points
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries


# Step 3 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries

# Step 4 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

# Step 5 - Form the feature and target vectors for training
featureTrainingVectors, targetTrainingVectors = util.formContinousFeatureAndTargetVectors(trainingSeries, depth)


# Step 6 - Train the network
util.trainESNWithoutTuning(size=1500, featureVectors=featureTrainingVectors, targetVectors=targetTrainingVectors,
                            initialTransient=50, inputConnectivity=0.7, reservoirConnectivity=0.9,
                            inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.20)
# util.trainESNWithMinimalTuning(size=1500,
#                                featureTrainingVectors=featureTrainingVectors,
#                                targetTrainingVectors=targetTrainingVectors,
#                                featureValidationVectors=featureValidationVectors,
#                                targetValidationVectors=targetValidationVectors,
#                                initialTransient=50)

# Step 7 - Predict the future
predictedSeries = util.predictFuture(trainingSeries, depth, horizon)

# Step 8 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)

# Step 9 - Plot the results
details = "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth)
util.plotSeries("Outputs/Outputs-Pandas_hourly" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Photo Interaction-Audi", "Photo Interaction (Likes+Comments+Shares)")
