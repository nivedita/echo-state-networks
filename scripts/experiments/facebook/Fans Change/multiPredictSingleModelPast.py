from utility import Utility
from datetime import datetime
import sys
from performance import ErrorMetrics as metrics
import pandas as pd
from reservoir import Utility as utilRes, classicESN as ESN, ReservoirTopology as topology, ActivationFunctions as act
import os
from plotting import OutputPlot as outputPlot
import numpy as np
from sklearn import preprocessing as pp

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, NuSVR


# Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "taylor_swift"
datasetFileName = directoryName + profileName + ".csv"


util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to daily)
resampledSeries = util.resampleSeriesSum(series, "D")
del series

# Step 3 - Scale the series
rawSeries = util.scaleSeriesStandard(resampledSeries)

# Step 4 - Remove the outliers
correctedSeries = util.detectAndRemoveOutliers(rawSeries)

# Learning Process - Start

# Parameters
depth = 60
horizon = 7

# Form feature and target vectors
featureVectors, targetVectors = util.formFeatureAndTargetVectorsMultiHorizon(correctedSeries, depth, horizon)


# One model for all range of horizons
model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])
model.fit(featureVectors, targetVectors)

# Predict
predictedTargetVectors = model.predict(featureVectors)

outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
for i in range(horizon):
    # Plot the actual and predicted
    actual = targetVectors[:, i]
    predicted = predictedTargetVectors[:, i]

    # Descale
    actual = util.scalingFunction.inverse_transform(actual)
    predicted = util.scalingFunction.inverse_transform(predicted)

    outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction_horizon"+str(i+1)+".html", "Facebook Fans Change - Linear Regression", "Taylor Swift", "Time", "Output")
    outplot.setXSeries(np.arange(1, targetVectors.shape[0]))
    outplot.setYSeries('Actual Output', actual)
    outplot.setYSeries('Predicted Output', predicted)
    outplot.createOutput()


print("Done!")













