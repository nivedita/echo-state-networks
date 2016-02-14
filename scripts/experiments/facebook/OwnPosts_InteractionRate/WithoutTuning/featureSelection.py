from utility import Utility
import sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
from plotting import OutputPlot as plot
from datetime import datetime
import os
from sklearn import preprocessing as pp

folderName = "Outputs/Outputs_" + str(datetime.now())
os.mkdir(folderName)
output = plot.OutputPlot(folderName+"/Correlation.html", "Correlation of output with depth", "", "Depth (hours)", "Correlation coefficient")

#Get the commamd line arguments
profileList = ["BMW", "Ferrari", "Dodge", "Jeep", "Mercedes-Benz"]

directoryName = "Datasets/"
for i in range(len(profileList)):

    profileName = profileList[i]
    datasetFileName = directoryName + profileName + "_time_interaction_rate.csv"

    daysOfHorizon = 10
    daysOfDepth = 90
    horizon = 24*daysOfHorizon#7 days ahead
    depth = 24*daysOfDepth #30 days
    util = Utility.SeriesUtility()

    series = util.convertDatasetsToSeries(datasetFileName)
    resampledSeries = util.resampleSeriesSum(series, "H")
    yearsOfData = 3
    recentCount = yearsOfData * 365 * 24 + horizon
    filteredSeries = util.filterRecent(resampledSeries, recentCount)

    normalizedSeries = util.scaleSeries(filteredSeries)
    trainingSeries, testingSeries = util.splitIntoTrainingAndTestingSeries(normalizedSeries,horizon)
    featureTrainingVectors, targetTrainingVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, depth)

    x_axis = []
    y_axis = []
    y = targetTrainingVectors[:,0]
    for i in range(featureTrainingVectors.shape[1]):
        #x_axis.append(depth - i)
        x_axis.append(i)
        x = featureTrainingVectors[:, i]

        correlation, p_value = pearsonr(x,y)
        y_axis.append(abs(correlation) )

    # Scale the correlation coefficient to 0.0 to 1.0
    y_axis = np.array(y_axis)
    scaler = pp.MinMaxScaler((0,1))
    y_axis = scaler.fit_transform(y_axis)

    output.setYSeries(profileName, y_axis)
output.setXSeries(np.array(x_axis))
output.createOutput()