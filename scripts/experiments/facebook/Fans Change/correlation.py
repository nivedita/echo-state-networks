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
depth = 100
featureVectors, targetVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(normalizedSeries, depth)

# Calculate the correlation
correlations = util.getRawCorrelationCoefficients(featureVectors, targetVectors)

folderName = "Outputs/Outputs_" + str(datetime.now())
os.mkdir(folderName)
output = outputPlot.OutputPlot(folderName+"/Correlation.html", "Correlation of output with depth", "", "Depth (days)", "Correlation coefficient")
output.setXSeries(np.arange(1, depth))
output.setYSeries("Correlations", correlations)
output.createOutput()


