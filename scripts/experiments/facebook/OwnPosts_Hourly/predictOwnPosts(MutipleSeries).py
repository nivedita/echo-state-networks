from utility import Utility
from datetime import datetime

# Dataset and profiles
datasetFileNames = ["facebookPosts_timestamp_audi_time.csv", "facebookPosts_timestamp_benz_time.csv",
                    "facebookPosts_timestamp_dodge_time.csv","facebookPosts_timestamp_ferrari_time.csv",
                    "facebookPosts_timestamp_jeep_time.csv"]
profileNames = ["Audi", "Benz", "Dodge", "Ferrari", "Jeep"]

# For plotting
seriesList = []
seriesNames = []

# Parameters for training
daysOfHorizon = 4
daysOfDepth = 30
horizon = 24*daysOfHorizon
depth = 24*daysOfDepth
yearsOfData = 5

# Util Object
globalUtil = Utility.SeriesUtility()


# Preprocess and sample the series( Series Intersection)
resampledSeries = []
for i in range(len(datasetFileNames)):
    # Convert the dataset into pandas series
    series = globalUtil.convertDatasetsToSeries(datasetFileNames[i])

    # Resample the series (to hourly)
    resampled = globalUtil.resampleSeries(series, "H")
    resampledSeries.append(resampled)
    del series

# Find intersection
processedSeries = globalUtil.intersect(resampledSeries)


for i in range(len(datasetFileNames)):

    print("Processing Profile " + str(i+1) + "..")

    util = Utility.SeriesUtility()

    # Todo - Train based on the recent data only.
    recentCount = yearsOfData * 365 * 24 + horizon #1 year of data for training+ horizon number of test data points
    filteredSeries = util.filterRecent(processedSeries[i], recentCount)

    # Step 3 - Scale the series
    normalizedSeries = util.scaleSeries(filteredSeries)
    del filteredSeries

    # Step 4 - Split into training and testing
    trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

    # Step 5 - Form the feature and target vectors for training
    featureVectors, targetVectors = util.formFeatureAndTargetVectors(trainingSeries, depth)

    # Step 6 - Train the network
    util.trainESNWithoutTuning(size=1500, featureVectors=featureVectors, targetVectors=targetVectors,
                                initialTransient=50, inputConnectivity=0.7, reservoirConnectivity=0.5,
                                inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3)
    #util.trainESNWithFullTuning(size=256, featureVectors=featureVectors, targetVectors=targetVectors, initialTransient=50)

    # Step 7 - Predict the future
    predictedSeries = util.predictFuture(trainingSeries, depth, horizon)

    # Step 8 - De-scale the series
    actualSeries = util.descaleSeries(testingSeries)
    predictedSeries = util.descaleSeries(predictedSeries)

    seriesNames.append("Actual_" + profileNames[i])
    seriesNames.append("Predicted_" + profileNames[i])
    seriesList.append(actualSeries)
    seriesList.append(predictedSeries)

# Step 9 - Plot the results
details = "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth)
globalUtil.plotSeries("Outputs/Outputs-Pandas_weekly_daily" + str(datetime.now()) + details,
                seriesList, seriesNames, "Facebook Own posts", "Number of posts")
