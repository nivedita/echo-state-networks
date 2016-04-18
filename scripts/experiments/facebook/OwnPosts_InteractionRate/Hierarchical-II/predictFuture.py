from utility import Utility
from datetime import datetime
import numpy as np
from reservoir import ActivationFunctions as activation, HierarchicalESNII as hesni

# Dataset
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

# Arbitrary Depth
arbitraryDepth = 24 * 60 # Depth of 1 year

# Constituent Network Parameters
lowerLayerParameters = {}
lowerLayerParameters['initialTransient'] = 0
lowerLayerParameters['spectralRadius'] = 0.79
lowerLayerParameters['inputScaling'] = 0.5
lowerLayerParameters['reservoirScaling'] = 0.5
lowerLayerParameters['leakingRate'] = 0.3
lowerLayerParameters['inputConnectivity'] = 1.0
lowerLayerParameters['reservoirConnectivity'] = 0.1
lowerLayerParameters['reservoirActivation'] = activation.LogisticFunction()
lowerLayerParameters['outputActivation'] = activation.ReLU()

# Composite Network Parameters
higherLayerParameters = {}
higherLayerParameters['initialTransient'] = 50
higherLayerParameters['spectralRadius'] = 0.79
higherLayerParameters['inputScaling'] = 0.5
higherLayerParameters['reservoirScaling'] = 0.5
higherLayerParameters['leakingRate'] = 0.3
higherLayerParameters['inputConnectivity'] = 1.0
higherLayerParameters['reservoirConnectivity'] = 0.1
higherLayerParameters['reservoirActivation'] = activation.LogisticFunction()
higherLayerParameters['outputActivation'] = activation.ReLU()

# Horizon - used to split the training and testing
daysOfHorizon = 14 # 10 days ahead
horizon = 24*daysOfHorizon
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Re-sample the series (to hourly)
resampledSeries = util.resampleSeriesSum(series, "H")
del series

# Step 3 - Filter recent data
yearsOfData = 3
recentCount = yearsOfData * 365 * 24 + horizon
filteredSeries = util.filterRecent(resampledSeries, recentCount)
del resampledSeries

# Step 4 - Scale the series
normalizedSeries = util.scaleSeries(filteredSeries)
del filteredSeries

# Step 5 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries, horizon)

# Step 6 - Form the feature and target vectors for training
trainingInputData, trainingOutputData = util.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, arbitraryDepth)

lowerLayerParameters['size'] = int(trainingInputData.shape[0]/10)
higherLayerParameters['size'] = int(trainingInputData.shape[0]/10)

# Feature Indices list for lower layer networks
# Lower layer network 1
# 24 hour activations
featureIndicesList = []
featureIndices = []
numberOfDays = 60
interval = 24
for i in range(numberOfDays - 1):
    featureIndices.append(interval*i)
featureIndices = np.array(featureIndices)
featureIndicesList.append(featureIndices)


# Step 7 - Train the hierarchical echo state network
hesn = hesni.HierarchicalESN(lowerLayerParameters=lowerLayerParameters,
                             higherLayerParameters=higherLayerParameters,
                             featureIndicesList=featureIndicesList,
                             inputData=trainingInputData,
                             outputData=trainingOutputData)
hesn.trainReservoir()


# Step 8 - Predict the future
predictedSeries = util.predictHierarchy(hesn, trainingSeries, arbitraryDepth, horizon, featureIndices)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)


# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon)
util.plotSeries("Outputs/Outputs_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
