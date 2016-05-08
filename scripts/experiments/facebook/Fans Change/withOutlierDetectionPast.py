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
depth = 100

# Form feature and target vectors
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(correctedSeries, depth)
featureVectors, targetVectors = util.formFeatureAndTargetVectors(correctedSeries, depth)


# # Train using linear regression
#model = SVR(kernel="linear")
model = NuSVR(nu=1.0, kernel="linear")
model.fit(featureVectors, targetVectors[:, 0])
predictedTrainingOutputData = model.predict(featureVectors)

targetVectors = targetVectors

# Predicted and actual Series
actualSeries = pd.Series(data=targetVectors.flatten(), index=correctedSeries.index[-targetVectors.shape[0]:])
predictedSeries = pd.Series(data=predictedTrainingOutputData.flatten(), index=correctedSeries.index[-targetVectors.shape[0]:])

# Learning Process - End

# Step 5 - Descale the series
actualSeries = util.descaleSeries(actualSeries)
predictedSeries = util.descaleSeries(predictedSeries)


outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
util.plotSeries(outputFolderName, [actualSeries, predictedSeries], ["Actual Series", "Predicted Series"], "Facebook Fans Change", "Outlier Detection")




