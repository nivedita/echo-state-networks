from utility import Utility
import numpy as np
from plotting import OutputTimeSeries as plot
from datetime import datetime
import os
import pandas as pd

#Get the commamd line arguments
directoryName = "Datasets/fans_change_"
profileName = "taylor_swift"
datasetFileName = directoryName + profileName + ".csv"

outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)


util = Utility.SeriesUtility()

# Step 1 - Convert the dataset into pandas series
series = util.convertDatasetsToSeries(datasetFileName)

# Step 2 - Resample the series (to daily)
resampledSeries = util.resampleSeriesExist(series, "D")
del series

# Step 3 - Find the mean and standard deviation
data = resampledSeries.values.flatten()


# Step 4 - Detect outliers
k = 3
mean = np.mean(data)
std = np.std(data)
deviation = abs(data - mean)
data[deviation > (k * std)] = 1
data[deviation <= (k * std)] = 0

# Actual Series
actualSeries = resampledSeries

# Outlier detected Series
outlierSeries = pd.Series(data=data, index=resampledSeries.index)

# Form x-axis
xAxis = []
color = []
radius = []
numberOfOUtliers = 0
for index, value in actualSeries.iteritems():
    year = index.strftime("%Y")
    month = index.strftime("%m")
    day = index.strftime("%d")
    hour = index.strftime("%H")
    nextDayStr = "Date.UTC(" + str(year)+","+ str(int(month)-1) + "," + str(day) + ")"
    xAxis.append(nextDayStr)

    if(outlierSeries[index] == 1):
        color.append("red")
        radius.append(10)
        numberOfOUtliers += 1
    else:
        color.append("blue")
        radius.append(4)

# Step 4 - Plot the actual data
fileName = "/Outlier_Detection_Using_STD.html"
actualDataPlot = plot.OutputTimeSeriesPlot(outputFolderName+fileName, "Facebook Fans Change", "Outlier Detection using standard deviation approach", "Fans Change")

actualDataPlot.setSeriesWithColorAndRadius("Fans Change", np.array(xAxis), actualSeries.values.flatten(), np.array(color), np.array(radius))
actualDataPlot.createOutput()


print("Number of outliers: "+str(numberOfOUtliers))





