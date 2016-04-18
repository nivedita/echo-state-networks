from utility import Utility
from datetime import datetime
import sys
from performance import ErrorMetrics as metrics


#Get the commamd line arguments
directoryName = "Datasets/"
profileName = "Jeep"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

daysOfHorizon = 21
daysOfDepth = 30
horizon = 24*daysOfHorizon#14 days ahead
depth = 24*daysOfDepth #30 days
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to hourly)
resampledSeries = util.resampleSeriesSum(series, "H")
del series

# Todo - Train based on the recent data only.
yearsOfData = 3
recentCount = yearsOfData * 365 * 24 + horizon
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries


# Step 3 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries

# Step 4 - Split into training, validation and testing series
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries, horizon)
availableSeries = trainingSeries
trainingSeries, validationSeries = util.splitIntoTrainingAndValidationSeries(trainingSeries, trainingSetRatio=0.95)


# Step 5 - Tune and Train the network
networkSize = 500
util.trainESNWithCorrelationTuned(size=networkSize,
                                  trainingSeries=trainingSeries,
                                  validationSeries=validationSeries,
                                  availableSeries=availableSeries,
                                  initialTransient=50,
                                  depth=depth
                                  )


# Step 6 - Predict the future
predictedSeries = util.predictFuture(availableSeries, depth, horizon)


# Step 7 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)

error = metrics.MeanSquareError()
regressionError = error.compute(testingSeries.values, predictedSeries.values)
print(regressionError)


# Step 8 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon) + "_depth_" + str(daysOfDepth) + "_network_size_" + str(networkSize)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
