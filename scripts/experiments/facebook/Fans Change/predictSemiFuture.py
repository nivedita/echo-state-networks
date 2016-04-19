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


#Get the commamd line arguments
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
normalizedSeries = util.scaleSeries(resampledSeries)
del resampledSeries

# Step 4 - Form feature and target vectors
depth = 60
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(normalizedSeries, depth)
n = featureVectors.shape[0]


# Divide the sets into two - one for training and one for testing
ratio = 0.9
cutOffPoint = int(ratio * featureVectors.shape[0])
trainingFeatureVectors = featureVectors[:cutOffPoint, :]
testingFeatureVectors = featureVectors[cutOffPoint:, :]
trainingTargetVectors = targetVectors[:cutOffPoint, :]
testingTargetVectors = targetVectors[cutOffPoint:, :]

# Train using linear regression
#model = Pipeline([('poly', PolynomialFeatures(degree=1)), ('Ridge', Ridge(fit_intercept=False, alpha=0.05))])
#model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])
model = SVR()
model.fit(trainingFeatureVectors, trainingTargetVectors[:, 0])
predictedTrainingOutputData = model.predict(testingFeatureVectors)

#
# error = metrics.RootMeanSquareError().compute(testingTargetVectors, predictedTrainingOutputData)
# print("Regression error:"+str(error))

#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Facebook Fans Change - Linear Regression", "Taylor Swift", "Time", "Output")
outplot.setXSeries(np.arange(1, n + 1))
outplot.setYSeries('Actual Output', testingTargetVectors[:,0])
outplot.setYSeries('Predicted Output', predictedTrainingOutputData)
outplot.createOutput()
print("Done!")



