from utility import Utility
from datetime import datetime
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np

# Dataset
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

# Horizon - used to split the training and testing
daysOfHorizon = 28 # 10 days ahead
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
featureIndices = []
numberOfDays = 60
interval = 24
arbitraryDepth = numberOfDays * interval
for i in range(numberOfDays - 1):
    featureIndices.append(interval*i)
featureIndices = np.array(featureIndices)

featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, arbitraryDepth)
featureVectors = featureVectors[:, featureIndices]
featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)),featureVectors))



# Step 7 - Train the network
networkSize = int(featureVectors.shape[0]/10)
util.trainESNWithoutTuning(size=networkSize, featureVectors=featureVectors, targetVectors=targetVectors,
                            initialTransient=50, inputConnectivity=1.0, reservoirConnectivity=0.1,
                            inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.30,
                            learningMethod=Utility.LearningMethod.Batch)


# Step 8 - Predict the future
predictedSeries = util.predict(util.esn, trainingSeries, arbitraryDepth, horizon, featureIndices)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)


# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) +  "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
