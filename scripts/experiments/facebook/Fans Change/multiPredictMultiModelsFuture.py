from utility import Utility
from datetime import datetime
import os
from plotting import OutputPlot as outputPlot
import numpy as np
import pandas as pd
from sklearn.svm import SVR, NuSVR
from reservoir import ReservoirTopology as topology, classicESN as ESN


# Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "taylor_swift"
datasetFileName = directoryName + profileName + ".csv"

# Parameters
depth = 60
horizon = 7

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

# Now, divide the series into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(correctedSeries, horizon)
availableSeries = trainingSeries

# Learning Process - Start

# Form feature and target vectors
featureVectors, targetVectors = util.formFeatureAndTargetVectorsMultiHorizon(trainingSeries, depth, horizon)

predictedSeries = []
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
for i in range(horizon):
    # Train different models for different horizon
    # Train the model
    model = NuSVR(kernel="rbf", gamma=1.0, nu=1.0, C=4, tol=1e-10)
    model.fit(featureVectors, targetVectors[:, i])

    # Now, predict the future
    featureVector = availableSeries.values[-depth:].reshape(1,depth)
    predicted = model.predict(featureVector)
    predictedSeries.append(predicted)

predictedSeries = pd.Series(data=np.array(predictedSeries).flatten(), index=testingSeries.index)

# Descale the series
predictedSeries = util.descaleSeries(predictedSeries)
actualSeries = util.descaleSeries(testingSeries)

# Plot the results
details = profileName + "_horizon_" + str(horizon) + "_depth_" + str(depth)
util.plotSeries("Outputs/Outputs_" + str(datetime.now()) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Fans Change - "+profileName, "Taylor Swift")
















