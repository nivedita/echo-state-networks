from utility import Utility


datasetFileName = "facebookPosts_timestamp_bmw_time.csv"
horizon = 24*4 #4 days ahead
depth = 24*30 #30 days
util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to hourly)
resampledSeries = util.resampleSeries(series, "H")
del series

# Step 3 - Scale the series
normalizedSeries = util.scaleSeries(resampledSeries)
del resampledSeries

# Step 4 - Split into training and testing
trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)

# Step 5 - Form the feature and target vectors for training
featureVectors, targetVectors = util.formFeatureAndTargetVectors(trainingSeries, depth)

# Step 6 - Train the network
util.trainESNWithoutTuning(size=1000, featureVectors=featureVectors, targetVectors=targetVectors,
                           initialTransient=50, inputConnectivity=0.7, reservoirConnectivity=0.5,
                           inputScaling=0.5, reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3)

# Step 7 - Predict the future
predictedSeries = util.predictFuture(trainingSeries, depth, horizon)

# Step 8 - De-scale the series
actualSeries = util.descaleSeries(testingSeries)
predictedSeries = util.descaleSeries(predictedSeries)

# Step 8 - Plot the results
