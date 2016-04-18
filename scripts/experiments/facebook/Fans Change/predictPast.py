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
depth = 30
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectors(normalizedSeries, depth)

n = featureVectors.shape[0]


# Input-to-reservoir fully connected
size = 1000
inputWeight = topology.ClassicInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size).generateWeightMatrix()

# Reservoir-to-reservoir fully connected
reservoirWeight = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()

res = ESN.Reservoir(size=size,
                    inputData=featureVectors,
                    outputData=targetVectors,
                    spectralRadius=1.1,
                    inputScaling=0.69,
                    reservoirScaling=0.31,
                    leakingRate=0.78,
                    initialTransient=0,
                    inputWeightRandom=inputWeight,
                    reservoirWeightRandom=reservoirWeight)
res.trainReservoir()

#Warm up
predictedTrainingOutputData = res.predict(featureVectors)


error = metrics.RootMeanSquareError().compute(targetVectors, predictedTrainingOutputData)
print("Regression error:"+str(error))

#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Facebook Fans Change - Classic ESN", "Taylor Swift", "Time", "Output")
outplot.setXSeries(np.arange(1, n + 1))
outplot.setYSeries('Actual Output', targetVectors[:,0])
outplot.setYSeries('Predicted Output', predictedTrainingOutputData[:, 0])
outplot.createOutput()
print("Done!")



