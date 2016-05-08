from utility import Utility
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.svm import SVR, NuSVR
from reservoir import ReservoirTopology as topology, classicESN as ESN, ActivationFunctions as act
from sklearn import preprocessing as pp

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
correctedSeries = util.scaleSeriesStandard(correctedSeries)


# Divide the series into training and testing series
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(correctedSeries, horizon)

# Learning Process - Start

# Form the feature and target vectors
featureVectors, targetVectors = formFeatureAndTargetVectors(trainingSeries)

# Fit a model
model = NuSVR(kernel="rbf", gamma=1.0, nu=1.0, tol=1e-15)
model.fit(featureVectors, targetVectors[:, 0])

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















