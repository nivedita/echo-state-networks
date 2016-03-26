from utility import Utility
from datetime import datetime
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np
from reservoir import ActivationFunctions as activation, HierarchicalESNI as hesni

# Dataset
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

# Arbitrary Depth
arbitraryDepth = 24 * 60 # Depth of 1 year

# Constituent Network Parameters
constituentNetworkParameters = {}
constituentNetworkParameters['size'] = 60
constituentNetworkParameters['initialTransient'] = 0
constituentNetworkParameters['spectralRadius'] = 0.79
constituentNetworkParameters['inputScaling'] = 0.5
constituentNetworkParameters['reservoirScaling'] = 0.5
constituentNetworkParameters['leakingRate'] = 0.3
constituentNetworkParameters['inputConnectivity'] = 0.1
constituentNetworkParameters['reservoirConnectivity'] = 0.3
constituentNetworkParameters['reservoirActivation'] = activation.LogisticFunction()
constituentNetworkParameters['outputActivation'] = activation.ReLU()

# Composite Network Parameters
compositeNetworkParameters = {}
compositeNetworkParameters['size'] = 1500
compositeNetworkParameters['initialTransient'] = 50
compositeNetworkParameters['spectralRadius'] = 0.79
compositeNetworkParameters['inputScaling'] = 0.5
compositeNetworkParameters['reservoirScaling'] = 0.5
compositeNetworkParameters['leakingRate'] = 0.3
compositeNetworkParameters['inputConnectivity'] = 1.0
compositeNetworkParameters['reservoirConnectivity'] = 0.4
compositeNetworkParameters['reservoirActivation'] = activation.LogisticFunction()
compositeNetworkParameters['outputActivation'] = activation.ReLU()

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

# Feature Indices
featureIndicesList = []

# Constituent Network 1
featureIndicesNetwork1 = np.arange(1, 60) * 24
featureIndicesList.append(featureIndicesNetwork1)

# Step 7 - Train the hierarchical echo state network
hesn = hesni.HierarchicalESN(constituentNetworkParameters=constituentNetworkParameters,
                             compositeNetworkParameters=compositeNetworkParameters,
                             featureIndicesList=featureIndicesList,
                             inputData=trainingInputData,
                             outputData=trainingOutputData)
hesn.trainReservoir()


# Step 8 - Predict


# Step 9 - Plot the results