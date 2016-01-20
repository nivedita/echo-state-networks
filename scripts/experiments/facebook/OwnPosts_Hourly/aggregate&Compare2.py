from utility import Utility
from datetime import datetime
import pandas as pd
import os
import numpy as np
from plotting import OutputTimeSeries as plotting

# Dataset and profiles
datasetFileNames = ["facebookPosts_timestamp_audi_time.csv", "facebookPosts_timestamp_benz_time.csv",
                    "facebookPosts_timestamp_dodge_time.csv","facebookPosts_timestamp_ferrari_time.csv"]
profileNames = ["Audi", "Benz", "Dodge", "Ferrari"]


#Perspective
perspective = "Audi"
competitors = ["Benz", "Dodge", "Ferrari"]

# For plotting
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
    resampled = globalUtil.resampleSeriesSum(series, "H")
    resampledSeries.append(resampled)
    del series

# Find intersection
processedSeries = globalUtil.intersect(resampledSeries)

# Aggregate the competitor series into one
processedSeriesNew = []
aggregatedCompetitor = pd.Series()
for i in range(len(datasetFileNames)):
    if profileNames[i] == perspective:
        processedSeriesNew.append(processedSeries[i])
    else:
        if(aggregatedCompetitor.empty):
            aggregatedCompetitor = processedSeries[i]
        else:
            aggregatedCompetitor = aggregatedCompetitor.add(processedSeries[i])
processedSeriesNew.append(aggregatedCompetitor)
processedSeries = processedSeriesNew
profileNames = ["Audi", "Competitors(Benz, Dodge, Ferrari)"]

# Start training for each series
for i in range(len(processedSeries)):

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
    featureVectors, targetVectors = util.formContinousFeatureAndTargetVectors(trainingSeries, depth)

    #Train the network
    util.trainESNWithoutTuning(size=1500, featureVectors=featureVectors, targetVectors=targetVectors,
                                initialTransient=50, inputConnectivity=0.9, reservoirConnectivity=0.8,
                                inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.60, leakingRate=0.45)
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

    seriesList.append(predictedSeries)


# Plot the results
details = "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth)
globalUtil.plotSeries("Outputs/Outputs-Pandas_hourly_aggregated" + str(datetime.now()) + details,
                seriesList, seriesNames, "Facebook Own posts - Aggregated", "Number of Posts")


