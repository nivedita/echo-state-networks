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
profileList = ["AcerDE"]

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

    spearman_axis = util.getRawCorrelationCoefficients(featureTrainingVectors, targetTrainingVectors)

    #output.setYSeries(profileName+"_pearson", pearson_axis)
    output.setYSeries(profileName, spearman_axis)
output.setXSeries(np.arange(depth, 0,-1))
output.createOutput()