import pandas as pd
import numpy as np
from plotting import OutputTimeSeries as plotting
import os
from datetime import datetime

def _sum(x):
    if len(x) == 0:
        return 0
    else:
        return sum(x)

# Read the data
df = pd.read_csv('facebookPosts_ferrari_time_photo_interaction.csv', index_col=0, parse_dates=True, names=['value'])

# Convert the dataframe into series
series = pd.Series(data=np.array(df.as_matrix()).flatten(), index=df.index)
resampled_series = series.resample(rule='H', how=_sum)

xAxis = []
yAxis = []
for index,value in resampled_series.iteritems():
    year = int(index.year)
    month = int(index.month)-1
    day = int(index.day)
    hour = int(index.hour)
    nextDayStr = "Date.UTC(" + str(year)+","+ str(int(month)) + "," + str(day) + ","+ str(hour)+")"
    xAxis.append(nextDayStr)
    yAxis.append(str(value))

# Plotting of the actual and prediction output
outputFolderName = "Outputs/Outputs-OwnPosts-Hourly" + str(datetime.now())
os.mkdir(outputFolderName)
outplot = plotting.OutputTimeSeriesPlot(outputFolderName + "/OwnPosts.html", "Facebook Own Posts Interaction-Ferrari", "", "Interaction")
outplot.setSeries('Own Posts', np.array(xAxis), np.array(yAxis))
outplot.createOutput()
