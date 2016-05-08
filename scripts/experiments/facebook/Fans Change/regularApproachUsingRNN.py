from utility import Utility
from datetime import datetime
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

# Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "Ferrari"
datasetFileName = directoryName + profileName + ".csv"
horizon = 7

def formFeatureAndTargetVectors(series):
    featureVectors = []
    targetVectors = []
    for index, value in series.iteritems():

        # Basic features
        year = index.year
        month = index.month
        day = index.day

        featureVectors.append(np.array([year, month, day]))
        targetVectors.append(value)

    featureVectors = np.array(featureVectors).reshape((len(featureVectors), 3))
    targetVectors = np.array(targetVectors).reshape((len(targetVectors), 1))
    return featureVectors, targetVectors

util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to daily)
resampledSeries = util.resampleSeriesSum(series, "D")
del series

# Step 5 - Remove the outliers
correctedSeries = util.detectAndRemoveOutliers(resampledSeries)

# Step 3 - Scale the series
correctedSeries = util.scaleSeries(correctedSeries)


# Divide the series into training and testing series
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(correctedSeries, horizon)

# Learning Process - Start

# Form the feature and target vectors
featureVectors, targetVectors = formFeatureAndTargetVectors(trainingSeries)

# Learning START
in_neurons = featureVectors.shape[1]
out_neurons = targetVectors.shape[1]
hidden_neurons = 100
batchSize = 1
epochs = 2000
model = Sequential()
model.add(Dense(hidden_neurons, input_dim=in_neurons))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="sgd")
model.fit(featureVectors, targetVectors,nb_epoch=epochs, validation_split=0.05)

# Learning Process - End

# Predict for testing data points
testingFeatureVectors, testingTargetVectors = formFeatureAndTargetVectors(testingSeries)
predictedTrainingOutputData = model.predict(testingFeatureVectors)

# Predicted and actual Series
actualSeries = testingSeries
predictedSeries = pd.Series(data=predictedTrainingOutputData.flatten(), index=testingSeries.index)

# Learning Process - End

# Step 5 - Descale the series
actualSeries = util.descaleSeries(actualSeries)
predictedSeries = util.descaleSeries(predictedSeries)


outputFolderName = "Outputs/"+str(profileName)+"Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
util.plotSeries(outputFolderName, [actualSeries, predictedSeries], ["Actual Series", "Predicted Series"], "Facebook Fans Change", "Outlier Detection")
