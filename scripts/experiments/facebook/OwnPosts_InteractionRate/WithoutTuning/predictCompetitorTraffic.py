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
    datasetFileName = directoryName + seriesName + "_time_interaction.csv"

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

    # Step 5 - Split into training and testing
    trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries, horizon)

    # Step 6 - Form the feature and target vectors for training
    featureIntervalList = []

    # # 24 hour interval
    period = 60
    for i in range(period, 0, -1):
        t = -24*i
        featureIntervalList.append(pd.Timedelta(hours=t))    # T

    targetIntervalList = [pd.Timedelta(hours=0)]


    featureTrainingVectors, targetTrainingVectors = tsi.TimeSeriesIntervalProcessor(trainingSeries, featureIntervalList, targetIntervalList).getProcessedData()
    featureTrainingVectors = np.hstack((np.ones((featureTrainingVectors.shape[0], 1)),featureTrainingVectors))


    # Step 7 - Train the network
    networkSize = int(featureTrainingVectors.shape[0]/10)
    util.trainESNWithoutTuning(size=networkSize, featureVectors=featureTrainingVectors, targetVectors=targetTrainingVectors,
                                initialTransient=50, inputConnectivity=1.0, reservoirConnectivity=0.3,
                                inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.30,
                                learningMethod=Utility.LearningMethod.Batch)


    # Step 8 - Predict the future
    predictedSeries = util.predictFutureWithFeatureInterval(trainingSeries, featureIntervalList, horizon)


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
    individualSeriesData.append(pd.Series(data=actual.values, index=actual.index))
    individuseriesNames.append(series+" Actual")
    individualSeriesData.append(pd.Series(data=predicted.values, index=predicted.index))
    individuseriesNames.append(series+" Predicted")

    if(competitorActual is None):
        competitorActual = pd.Series(data=actual.values, index=actual.index)
        competitorPredicted = pd.Series(data=predicted.values, index=predicted.index)
    else:
        competitorActual+= actual
        competitorPredicted+= predicted

competitorSeriesData.append(competitorActual)
competitorSeriesNames.append("Competitors Actual")
competitorSeriesData.append(competitorPredicted)
competitorSeriesNames.append("Competitors Predicted")


# Remove Nan's
for i in range(len(individualSeriesData)):
    individualSeriesData[i].fillna(0.0, inplace=True)

for i in range(len(competitorSeriesData)):
    competitorSeriesData[i].fillna(0.0, inplace=True)

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
