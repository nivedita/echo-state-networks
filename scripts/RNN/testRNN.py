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

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 100

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
print(X_train.shape)
model.fit(X_train, y_train, nb_epoch=10, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print(rmse)

# Plot the results
#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction_1.html", "Mackey-Glass Time Series - Classic ESN", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, y_test.shape[0]))
outplot.setYSeries('Actual Output', y_test[:, 0])
outplot.setYSeries('Predicted Output', predicted[:, 0])
outplot.createOutput()

outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction_2.html", "Mackey-Glass Time Series - Classic ESN", "Prediction of future values", "Time", "Output")
outplot.setXSeries(np.arange(1, y_test.shape[0]))
outplot.setYSeries('Actual Output', y_test[:, 1])
outplot.setYSeries('Predicted Output', predicted[:, 1])
outplot.createOutput()
print("Done!")
