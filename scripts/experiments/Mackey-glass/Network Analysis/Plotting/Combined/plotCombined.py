from reservoir import GAUtility as utilityGA
from plotting import ScatterPlot as plot
import numpy as np
import os
from datetime import datetime


# File name for
folderName = "Outputs/GAResults_Combined_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(folderName)

# Load the best population from the file
bestPopulationErdos = utilityGA.loadBestPopulation("erdos.pkl")
bestPopulationScaleFree = utilityGA.loadBestPopulation("scalefree.pkl")
bestPopulationSmallWorld = utilityGA.loadBestPopulation("smallworld.pkl")
bestPopulationList = [bestPopulationErdos, bestPopulationScaleFree, bestPopulationSmallWorld]


# Iterate over all the elements and get the network properties
networkSize = 500
errorList = []
averageDegreeList = []
averagePathLengthList = []
averageDiameterList = []
averageClusteringCoefficientList = []

for bestPopulation in bestPopulationList:

    for item in bestPopulation:
        # Fitness
        attachment = item[0][0,0]/networkSize
        error = item[1]

        # Network properties
        averageDegree = item[2]['averageDegree']
        averagePathLength = item[2]['averagePathLength']
        averageDiameter = item[2]['averageDiameter']
        averageClusteringCoefficient = item[2]['averageClusteringCoefficient'] * 500

        # Append it the list
        errorList.append(error)
        averageDegreeList.append(averageDegree)
        averagePathLengthList.append(averagePathLength)
        averageDiameterList.append(averageDiameter)
        averageClusteringCoefficientList.append(averageClusteringCoefficient)

errorList = np.array(errorList).reshape(len(errorList),1)[:,0]
averageDegreeList = np.array(averageDegreeList).reshape(len(averageDegreeList),1)[:,0]
averagePathLengthList = np.array(averagePathLengthList).reshape(len(averagePathLengthList),1)[:,0]
averageDiameterList = np.array(averageDiameterList).reshape(len(averageDiameterList),1)[:,0]
averageClusteringCoefficientList = np.array(averageClusteringCoefficientList).reshape(len(averageClusteringCoefficientList),1)[:,0]


# Plot 2 - Average degree vs error
fileName = "/AverageDegree_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "All Networks Optimization using GA", "Average Degree vs Performance", "Average Degree", "Mean Square Error")
scatter.setSeries("Network Performance", averageDegreeList, errorList)
scatter.createOutput()


# Plot 3 - Average path length vs error
fileName = "/AveragePathLength_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "All Networks Optimization using GA", "Average Path Length vs Performance", "Average Path Length", "Mean Square Error")
scatter.setSeries("Network Performance", averagePathLengthList, errorList)
scatter.createOutput()

# Plot 4 - Diameter vs error
fileName = "/Diameter_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "All Networks Optimization using GA", "Diameter vs Performance", "Diameter", "Mean Square Error")
scatter.setSeries("Network Performance", averageDiameterList, errorList)
scatter.createOutput()

# Plot 5 - Average clustering coefficient vs error
fileName = "/AverageClusteringCoefficient_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "All Networks Optimization using GA", "Average Clustering Coefficient vs Performance", "Average Clustering Coefficient", "Mean Square Error")
scatter.setSeries("Network Performance", averageClusteringCoefficientList, errorList)
scatter.createOutput()

# Plot 6 - Histograms
utilityGA.plotHistogram(averageDegreeList, 5, (0,1), folderName+"/AverageDegreeHistogram.html", "Histogram - Average Degree", "for the best found in GA", "Average Degree", "Count", "Histogram")
utilityGA.plotHistogram(averagePathLengthList, 5, (0,1), folderName+"/AveragePathLengthHistogram.html", "Histogram - Average Path Length", "for the best found in GA", "Average Path Length", "Count", "Histogram")
utilityGA.plotHistogram(averageDiameterList, 5, (0,1), folderName+"/DiameterHistogram.html", "Histogram - Diameter", "for the best found in GA", "Diameter", "Count", "Histogram")
utilityGA.plotHistogram(averageClusteringCoefficientList, 5, (0,1), folderName+"/AverageCCHistogram.html", "Histogram - Average Clustering Coefficient", "for the best found in GA", "Average Clustering Coefficient", "Count", "Histogram")

