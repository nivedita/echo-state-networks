from utility import Utility
from datetime import datetime
import numpy as np
from reservoir import ActivationFunctions as activation, HierarchicalESNI as hesni

# Dataset
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

# Arbitrary Depth
arbitraryDepth = 24 * 60 # Depth of 1 year

# Constituent Network Parameters
featureTransformerParameters = {}
featureTransformerParameters['initialTransient'] = 0
featureTransformerParameters['spectralRadius'] = 0.79
featureTransformerParameters['inputScaling'] = 0.5
featureTransformerParameters['reservoirScaling'] = 0.5
featureTransformerParameters['leakingRate'] = 1.0
featureTransformerParameters['inputConnectivity'] = 1.0
featureTransformerParameters['reservoirConnectivity'] = 0.8
featureTransformerParameters['reservoirActivation'] = activation.LogisticFunction()
featureTransformerParameters['outputActivation'] = activation.ReLU()

# Composite Network Parameters
predictorParameters = {}
predictorParameters['initialTransient'] = 50
predictorParameters['spectralRadius'] = 0.79
predictorParameters['inputScaling'] = 0.5
predictorParameters['reservoirScaling'] = 0.5
predictorParameters['leakingRate'] = 0.3
predictorParameters['inputConnectivity'] = 1.0
predictorParameters['reservoirConnectivity'] = 0.3
predictorParameters['reservoirActivation'] = activation.LogisticFunction()
predictorParameters['outputActivation'] = activation.ReLU()

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

featureTransformerParameters['size'] = 60
predictorParameters['size'] = int(trainingInputData.shape[0]/10)


# Step 7 - Train the hierarchical echo state network
hesn = hesni.HierarchicalESN(featureTransformerParameters=featureTransformerParameters,
                             predictorParameters=predictorParameters,
                             inputData=trainingInputData,
                             outputData=trainingOutputData,
                             depth=arbitraryDepth)
hesn.trainReservoir()


# Step 8 - Predict the future
predictedSeries = util.predictFutureFeatureTransformer(hesn, trainingSeries, arbitraryDepth, horizon)


# Step 9 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)


# Step 10 - Plot the results
details = profileName + "_yearsOfData_" + str(yearsOfData) + "_horizon_" + str(daysOfHorizon)
util.plotSeries("Outputs/Outputs_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + details,
                [actualSeries, predictedSeries], ["Actual Output", "Predicted Output"], "Facebook Own Posts Interaction Rate - "+profileName, "Interaction Rate")
