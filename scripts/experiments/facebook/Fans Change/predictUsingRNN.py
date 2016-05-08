from utility import Utility
from datetime import datetime
import sys
from performance import ErrorMetrics as metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from timeseries import TimeSeries as ts
import pandas as pd

def formFeatureTargetVectors(data, depth=1, horizon=1):
    # Pre-process the data and form feature and target vectors
    tsp = ts.TimeSeriesProcessor(data, depth, horizon)
    featureVectors, targetVectors = tsp.getProcessedData()

    featureVectors = featureVectors.reshape(featureVectors.shape[0], depth, 1)
    return featureVectors, targetVectors

def predictFuture(network, availableSeries, depth, horizon):
    # To avoid mutation of pandas series
    initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
    for i in range(horizon):
        feature = initialSeries.values[-depth:].reshape((1, depth, 1))

        nextPoint = network.predict(feature)[0,0]

        nextIndex = initialSeries.last_valid_index() + pd.Timedelta(days=1)
        initialSeries[nextIndex] = nextPoint

    predictedSeries = initialSeries[-horizon:]
    return predictedSeries

# Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "taylor_swift"
datasetFileName = directoryName + profileName + ".csv"

# Forecasting parameters
depth = 7
horizon = 7

# Network parameters
in_out_neurons = 1
hidden_neurons = 50
batch_size = 1

util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to daily)
resampledSeries = util.resampleSeriesSum(series, "D")
del series

# Remove the outliers
resampledSeries = util.detectAndRemoveOutliers(resampledSeries)

# Step 3 - Rescale the series
normalizedSeries = util.scaleSeriesStandard(resampledSeries)
del resampledSeries

# Step 4 - Split the data into training and testing series
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries, horizon)
availableSeries = trainingSeries

# Split the training into training and validation
trainingSeries, validationSeries = util.splitIntoTrainingAndValidationSeries(trainingSeries, 0.8)

# Step 4 - Form feature and target vectors
trainingFeatureVectors, trainingTargetVectors = formFeatureTargetVectors(trainingSeries, depth)
validationFeatureVectors, validationTargetVectors = formFeatureTargetVectors(validationSeries, depth)

# Stack the layers
model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False, stateful=True, batch_input_shape=(batch_size, depth, 1)))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(trainingFeatureVectors, trainingTargetVectors, nb_epoch=100, batch_size=batch_size, validation_data=(validationFeatureVectors, validationTargetVectors))
# Learning END

# Predict the future
predictedTestingOutputData = predictFuture(model, availableSeries, depth, horizon)

# Step 8 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedTestingOutputData)

# Step 9 - Plot the results
details = profileName + "_horizon_" + str(horizon) + "_depth_" + str(depth)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Fans Change - "+profileName, "Fans Change")




