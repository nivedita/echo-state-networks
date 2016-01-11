from utility import Utility
from datetime import datetime
import pandas as pd

# Dataset and profiles
datasetFileNames = ["facebookPosts_timestamp_audi_time.csv", "facebookPosts_timestamp_benz_time.csv",
                    "facebookPosts_timestamp_dodge_time.csv","facebookPosts_timestamp_ferrari_time.csv"]
profileNames = ["Audi", "Benz", "Dodge", "Ferrari"]


#Perspective
perspective = "Audi"
competitors = ["Benz", "Dodge", "Ferrari"]

# For plotting
actualPerspective = None
predictedPerspective = None
actualCompetitor = pd.Series()
predictedCompetitor = pd.Series()
seriesNames = []
seriesList = []

# Parameters for training
daysOfHorizon = 4
daysOfDepth = 45
horizon = 24*daysOfHorizon#7 days ahead
depth = 24*daysOfDepth #30 days
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

    # Scale the series
    normalizedSeries = util.scaleSeries(filteredSeries)
    del filteredSeries

    # Split into training and testing
    trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

    # Form the feature and target vectors for training
    featureVectors, targetVectors = util.formFeatureAndTargetVectors(trainingSeries, depth)

    #Train the network
    util.trainESNWithoutTuning(size=1500, featureVectors=featureVectors, targetVectors=targetVectors,
                                initialTransient=50, inputConnectivity=0.9, reservoirConnectivity=0.8,
                                inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.45)
    #util.trainESNWithFullTuning(size=1000, featureVectors=featureVectors, targetVectors=targetVectors, initialTransient=50)

    # util.trainESNWithMinimalTuning(size=500, featureVectors=featureVectors, targetVectors=targetVectors,
    #                                initialTransient=50, inputConnectivity=0.9, reservoirConnectivity=0.6)



    # Predict the future
    predictedSeries = util.predictFuture(trainingSeries, depth, horizon)

    # De-scale the series
    actualSeries = util.descaleSeries(testingSeries)
    predictedSeries = util.descaleSeries(predictedSeries)

    seriesNames.append("Actual_" + profileNames[i])
    seriesNames.append("Predicted_" + profileNames[i])
    seriesList.append(actualSeries)

    #Re-scale to fit the actual - This is a kind of trick to show a better learning
    tempUtil = Utility.SeriesUtility()
    predictedSeries = tempUtil.scaleSeries(predictedSeries)
    seriesList.append(predictedSeries)

    # Aggregator
    if(profileNames[i] == perspective):  # Perspective
        actualPerspective = actualSeries
        predictedPerspective = predictedSeries
    else: # Competitor
        if(actualCompetitor.empty):  # Empty
            actualCompetitor = actualSeries
            predictedCompetitor = predictedSeries
        else: # Not empty
            actualCompetitor = actualCompetitor.add(actualSeries)
            predictedCompetitor = predictedCompetitor.add(predictedSeries)

# Average the results for competitor
actualCompetitor = actualCompetitor.divide(len(competitors))
predictedCompetitor = predictedCompetitor.divide(len(competitors))

# Plot the results - aggregated
seriesList.append(actualCompetitor)
seriesList.append(predictedCompetitor)
seriesNames.append("Actual Competitors")
seriesNames.append("Predicted Competitors")
details = "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth)
globalUtil.plotSeries("Outputs/Outputs-Pandas_hourly_aggregated" + str(datetime.now()) + details,
                seriesList, seriesNames, "Facebook Own posts - Aggregated", "Posting Probability")


