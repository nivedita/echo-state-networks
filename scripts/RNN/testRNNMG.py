import pandas as pd
from random import random
from plotting import OutputPlot as plot
import os
from plotting import OutputPlot as outputPlot
from datetime import datetime
flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from timeseries import TimeSeries as ts
from reservoir import Utility as util
from sklearn import preprocessing as pp


def formFeatureTargetVectors(data, depth=1, horizon=1):
    # Pre-process the data and form feature and target vectors
    tsp = ts.TimeSeriesProcessor(data, depth, horizon)
    featureVectors, targetVectors = tsp.getProcessedData()

    featureVectors = featureVectors.reshape(featureVectors.shape[0], depth, 1)
    return featureVectors, targetVectors

def predictFuture(network, availableData, depth, horizon):
    # Predict future values
    availableList = list(availableData.flatten())
    for i in range(horizon):

        # Reset the model
        network.reset_states()

        # Form the feature
        feature = np.array(availableList)[-depth:].reshape(1, depth, 1)

        # Predict the next point
        nextPoint = network.predict(feature)[0,0]
        availableList.append(nextPoint)

    availableList = np.array(availableList)[-horizon:].reshape((horizon, 1))
    return availableList


# Forecasting parameters
depth = 30

# Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
#minMax = pp.MinMaxScaler((-1,1))
minMax = pp.StandardScaler()
data = minMax.fit_transform(data)

# Get only 6000 points
data = data[:5000].reshape((5000, 1))

# Number of points - 5000
trainingData, testingData = util.splitData2(data, 0.85)
availableData = trainingData
nTesting = testingData.shape[0]

# Divide the training data into training and validation
validationRatio = 0.4
trainingData, validationData = util.splitData2(trainingData, 1.0-validationRatio)

# Form feature vectors
trainingFeatureVectors, trainingTargetVectors = formFeatureTargetVectors(trainingData, depth)
validationFeatureVectors, validationTargetVectors = formFeatureTargetVectors(validationData, depth)
testingFeatureVectors, testingTargetVectors = formFeatureTargetVectors(testingData, depth)

# Network parameters
in_out_neurons = 1
hidden_neurons = 200
batch_size = 1

# Stack the layers
model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(trainingFeatureVectors, trainingTargetVectors, nb_epoch=30, validation_split=0.05)

testingPredicted = predictFuture(model, availableData, depth, nTesting)
testingActual = testingData


# Plot the results
#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series - Classic ESN", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', minMax.inverse_transform(testingActual[:nTesting, 0]))
outplot.setYSeries('Predicted Output', minMax.inverse_transform(testingPredicted[:nTesting, 0]))
outplot.createOutput()
print("Done!")
