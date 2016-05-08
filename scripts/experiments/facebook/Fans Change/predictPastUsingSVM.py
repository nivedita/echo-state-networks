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
from sklearn.svm import SVR


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

# Step 3 - Rescale the series
normalizedSeries = util.scaleSeriesStandard(resampledSeries)
actualSeries = normalizedSeries
del resampledSeries

# Step 4 - Remove the outliers
normalizedSeries = util.detectAndRemoveOutliers(normalizedSeries)


# Step 4 - Form feature and target vectors
depth = 100
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(normalizedSeries, depth)
n = featureVectors.shape[0]


# Train using linear regression
#model = SVR()
model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model.fit(featureVectors, targetVectors[:, 0])
predictedTrainingOutputData = model.predict(featureVectors)

# Descale
predictedTrainingOutputData = util.scalingFunction.inverse_transform(predictedTrainingOutputData)
targetVectors = util.scalingFunction.inverse_transform(targetVectors)

#targetVectors =actualSeries.values.reshape(actualSeries.values.shape[0],1)[depth:]

error = metrics.RootMeanSquareError().compute(targetVectors, predictedTrainingOutputData.reshape(predictedTrainingOutputData.shape[0],1))
print("Regression error:"+str(error))


#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Facebook Fans Change - Linear Regression", "Taylor Swift", "Time", "Output")
outplot.setXSeries(np.arange(1, n + 1))
outplot.setYSeries('Actual Output', targetVectors[:,0])
outplot.setYSeries('Predicted Output', predictedTrainingOutputData)
outplot.createOutput()
print("Done!")



