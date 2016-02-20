from utility import Utility
from datetime import datetime
from performance import ErrorMetrics as metrics
import pandas as pd
from timeseries import TimeSeriesInterval as tsi
import numpy as np

# Dataset
directoryName = "Datasets/"
profileName = "BMW"
datasetFileName = directoryName + profileName + "_time_interaction.csv"

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

data = normalizedSeries.values.flatten()

cumChange = 0.0
numChanges = 0
for i in range(1, data.shape[0]):

    previousValue = data[i-1]
    currentValue = data[i]
    change= abs(currentValue - previousValue)
    if(change > 0.0):
        numChanges = numChanges + 1
    cumChange += change

leakingRate = cumChange/ numChanges
print(leakingRate)
