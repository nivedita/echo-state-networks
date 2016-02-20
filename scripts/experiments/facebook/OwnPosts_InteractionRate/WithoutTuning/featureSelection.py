from utility import Utility
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
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
    datasetFileName = directoryName + profileName + "_time_interaction.csv"

    util = Utility.SeriesUtility()
    series = util.convertDatasetsToSeries(datasetFileName)
    resampledSeries = util.resampleSeriesSum(series, "H")
    yearsOfData = 3
    recentCount = yearsOfData * 365 * 24
    filteredSeries = util.filterRecent(resampledSeries, recentCount)
    depth = 365 * 24 #1 year
    normalizedSeries = util.scaleSeries(filteredSeries)
    featureTrainingVectors, targetTrainingVectors = util.formContinousFeatureAndTargetVectorsWithoutBias(normalizedSeries, depth)

    x_axis = []
    pearson_axis = []
    spearman_axis = []
    y = targetTrainingVectors[:,0]
    for i in range(featureTrainingVectors.shape[1]):
        x_axis.append(depth - i)
        x = featureTrainingVectors[:, i]

        # pearson_correlation, p_value = pearsonr(x,y)
        # pearson_axis.append(abs(pearson_correlation))

        spearman_correlation, p_value = spearmanr(x,y)
        spearman_axis.append(abs(spearman_correlation))

    # Scale the correlation coefficient to 0.0 to 1.0
    #pearson_axis = np.array(pearson_axis)
    spearman_axis = np.array(spearman_axis)
    #scaler_pearson = pp.MinMaxScaler((0,1))
    #pearson_axis = scaler_pearson.fit_transform(pearson_axis)
    #scaler_spearman = pp.MinMaxScaler((0,1))
    #spearman_axis = scaler_spearman.fit_transform(spearman_axis)

    #output.setYSeries(profileName+"_pearson", pearson_axis)
    output.setYSeries(profileName, spearman_axis)
output.setXSeries(np.array(x_axis))
output.createOutput()