from utility import Utility
from datetime import datetime
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np

# Reservoir Learning Parameters
networkParameters = {}
networkParameters['initialTransient'] = 50
networkParameters['spectralRadius'] = 0.79
networkParameters['inputScaling'] = 0.5
networkParameters['reservoirScaling'] = 0.5
networkParameters['leakingRate'] = 0.3
networkParameters['inputConnectivity'] = 1.0
networkParameters['reservoirConnectivity'] = 0.1

# Horizon - used to split the training and testing
daysOfHorizon = 14 # 10 days ahead
horizon = 24*daysOfHorizon
yearsOfData = 3

def predictSeries(seriesName):
    # Dataset
    directoryName = "Datasets/"
    profileName = seriesName
    datasetFileName = directoryName + profileName + "_time_interaction.csv"

    util = Utility.SeriesUtility()

    # Step 1 - Convert the dataset into pandas series
    series = util.convertDatasetsToSeries(datasetFileName)

    # Step 2 - Re-sample the series (to hourly)
    resampledSeries = util.resampleSeriesSum(series, "H")
    del series

    # Step 3 - Filter recent data
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
                               spectralRadius=networkParameters['spectralRadius'], leakingRate=optimalLeakingRate)


    # Step 8 - Predict the future
    predictedSeries = util.predict(util.esn, availableSeries, bestDepth, horizon, bestFeaturesIndices)


    # Step 9 - De-scale the series
    actualSeries = util.descaleSeries(testingSeries)
    predictedSeries = util.descaleSeries(predictedSeries)

    return actualSeries, predictedSeries


# List to hold series data for plotting
individualSeriesData = []
individuseriesNames = []
competitorSeriesData = []
competitorSeriesNames = []

# Predict for perspective profile
perspective = "Ferrari"
perspectiveActual, perspectivePredicted = predictSeries(perspective)
individualSeriesData.append(perspectiveActual)
individuseriesNames.append(perspective+" Actual")
individualSeriesData.append(perspectivePredicted)
individuseriesNames.append(perspective+" Predicted")

# competitorSeriesData.append(perspectiveActual)
# competitorSeriesNames.append(perspective+ " Actual")
# competitorSeriesData.append(perspectivePredicted)
# competitorSeriesNames.append(perspective+" Predicted")


# Predict for competitor profile
competitors = ['BMW', 'Mercedes-Benz']
competitorActual = None
competitorPredicted = None
for series in competitors:
    actual, predicted = predictSeries(series)
    individualSeriesData.append(actual)
    individuseriesNames.append(series+" Actual")
    individualSeriesData.append(predicted)
    individuseriesNames.append(series+" Predicted")

    if(competitorActual is None):
        competitorActual = actual
        competitorPredicted = predicted
    else:
        competitorActual+= actual
        competitorPredicted+= predicted

competitorSeriesData.append(competitorActual)
competitorSeriesNames.append("Competitors Actual")
competitorSeriesData.append(competitorPredicted)
competitorSeriesNames.append("Competitors Predicted")

# Plot the results - for individial profiles
util = Utility.SeriesUtility()
details = "Competitor Analysis - Individual" + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon)
util.plotSeries("Outputs/Outputs_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + details,
                individualSeriesData, individuseriesNames, "Facebook Own Posts Interaction Rate", "Interaction Rate")

# Plot the results for competitor traffic
util = Utility.SeriesUtility()
details = "Competitor Analysis - Aggregated" + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon)
util.plotCombinedSeries("Outputs/Outputs_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + details,
                competitorSeriesData, competitorSeriesNames, "Competitor's Upcoming Post Traffic for "+perspective, "", "Interaction Rate")
