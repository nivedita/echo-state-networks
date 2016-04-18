from utility import Utility
from datetime import datetime
from performance import ErrorMetrics as metrics
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np

# Dataset
directoryName = "Datasets/"
profileName = "Kia"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

# Horizon - used to split the training and testing
daysOfHorizon = 14 # 10 days ahead
horizon = 24*daysOfHorizon
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

#featureIntervalList = util.getBestFeatures(trainingSeries, 0.30)

targetIntervalList = [pd.Timedelta(hours=0)]


featureTrainingVectors, targetTrainingVectors = tsi.TimeSeriesIntervalProcessor(trainingSeries, featureIntervalList, targetIntervalList).getProcessedData()
featureTrainingVectors = np.hstack((np.ones((featureTrainingVectors.shape[0], 1)),featureTrainingVectors))


# Step 7 - Train the network
networkSize = int(featureTrainingVectors.shape[0]/10)
#networkSize = 1000
util.trainESNWithoutTuning(size=networkSize, featureVectors=featureTrainingVectors, targetVectors=targetTrainingVectors,
                            initialTransient=50, inputConnectivity=1.0, reservoirConnectivity=0.3,
                            inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.26)


# Step 8 - Predict the future
predictedSeries = util.predictFutureWithFeatureInterval(trainingSeries, featureIntervalList, horizon)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)

# Supress the negative values
#predictedSeries = util.removeNegativeValues(predictedSeries)

# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) +  "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
