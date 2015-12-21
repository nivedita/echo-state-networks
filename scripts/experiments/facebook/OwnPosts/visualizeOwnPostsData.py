import numpy as np
import os
from plotting import OutputTimeSeries as outTimePlot
from datetime import datetime

# Read data from the file
rawData = np.loadtxt("facebookPostsCount_bmw_raw.txt", delimiter=',')

actualData = rawData[:, 3]

xAxis = []
for i in range(rawData.shape[0]):
    year = int(rawData[i, 0])
    month = int(rawData[i, 1])
    day = int(rawData[i, 2])
    nextDayStr = "Date.UTC(" + str(year)+","+ str((int(month)-1)) + "," + str(day) +")"
    xAxis.append(nextDayStr)

# Plotting of the actual and prediction output
outputFolderName = "Outputs/Outputs OwnPosts" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = outTimePlot.OutputTimeSeriesPlot(outputFolderName + "/OwnPosts.html", "Facebook Own posts-BMW", "", "Number of posts")
outplot.setSeries('Own Posts', np.array(xAxis), actualData)
outplot.createOutput()
