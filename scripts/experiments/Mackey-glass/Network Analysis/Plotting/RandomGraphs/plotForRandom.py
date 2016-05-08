from reservoir import GAUtility as utilityGA
from plotting import ScatterPlot as plot
import numpy as np
import os
from datetime import datetime


# File name for
folderName = "Outputs/GAResults_Random_Graphs_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(folderName)
# Load the best population from the file
bestPopulation = utilityGA.loadBestPopulation("population.pkl")


# Iterate over all the elements and get the network properties
networkSize = 500
connectivityList = []
errorList = []
averageDegreeList = []
averagePathLengthList = []
averageDiameterList = []
averageClusteringCoefficientList = []
for item in bestPopulation:
    # Fitness
    connectivity = item[0][0,0]
    error = item[1]

    # Network properties
    averageDegree = item[2]['averageDegree']
    averagePathLength = item[2]['averagePathLength']
    averageDiameter = item[2]['averageDiameter']
    averageClusteringCoefficient = item[2]['averageClusteringCoefficient'] * 500

    # Append it the list
    connectivityList.append(connectivity)
    errorList.append(error)
    averageDegreeList.append(averageDegree)
    averagePathLengthList.append(averagePathLength)
    averageDiameterList.append(averageDiameter)
    averageClusteringCoefficientList.append(averageClusteringCoefficient)


attachmentList = np.array(connectivityList).reshape(len(connectivityList),1)[:,0]
errorList = np.array(errorList).reshape(len(errorList),1)[:,0]
averageDegreeList = np.array(averageDegreeList).reshape(len(averageDegreeList),1)[:,0]
averagePathLengthList = np.array(averagePathLengthList).reshape(len(averagePathLengthList),1)[:,0]
averageDiameterList = np.array(averageDiameterList).reshape(len(averageDiameterList),1)[:,0]
averageClusteringCoefficientList = np.array(averageClusteringCoefficientList).reshape(len(averageClusteringCoefficientList),1)[:,0]


# Plot 1 - Network parameters vs error
fileName = "/Connecitivity_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "Random Graphs Optimization using GA", "Connectivity vs Performance", "Connectivity", "Mean Square Error")
scatter.setSeries("Network Performance", attachmentList, errorList)
scatter.createOutput()


# Plot 2 - Average degree vs error
fileName = "/AverageDegree_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "Random Graphs Optimization using GA", "Average Degree vs Performance", "Average Degree", "Mean Square Error")
scatter.setSeries("Network Performance", averageDegreeList, errorList)
scatter.createOutput()


# Plot 3 - Average path length vs error
fileName = "/AveragePathLength_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "Random Graphs Optimization using GA", "Average Path Length vs Performance", "Average Path Length", "Mean Square Error")
scatter.setSeries("Network Performance", averagePathLengthList, errorList)
scatter.createOutput()

# Plot 4 - Average diameter vs error
fileName = "/AverageDiameter_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "Random Graphs Optimization using GA", "Average Diameter vs Performance", "Average Diameter", "Mean Square Error")
scatter.setSeries("Network Performance", averageDiameterList, errorList)
scatter.createOutput()

# Plot 5 - Average clustering coefficient vs error
fileName = "/AverageClusteringCoefficient_Vs_Error.html"
scatter = plot.ScatterPlot(folderName+fileName, "Random Graphs Optimization using GA", "Average Clustering Coefficient vs Performance", "Average Clustering Coefficient", "Mean Square Error")
scatter.setSeries("Network Performance", averageClusteringCoefficientList, errorList)
scatter.createOutput()
