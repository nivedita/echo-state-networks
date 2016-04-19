from reservoir import GAUtility as utilityGA
from plotting import ScatterPlot as plot
import numpy as np


# File name for
folderName = "Outputs/GAResults_Scale_Free_Networks/"

# Load the best population from the file
bestPopulation = utilityGA.loadBestPopulation("population.pkl")


# Iterate over all the elements and get the network properties
attachmentList = []
errorList = []
averageDegreeList = []
averagePathLengthList = []
averageDiameterList = []
averageClusteringCoefficientList = []
for item in bestPopulation:
    # Fitness
    attachment = item[0]
    error = item[1]

    # Network properties
    averageDegree = item[2]['averageDegree']
    averagePathLength = item[2]['averagePathLength']
    averageDiameter = item[2]['averageDiameter']
    averageClusteringCoefficient = item[2]['averageClusteringCoefficient']

    # Append it the list
    averageDegreeList.append(averageDegree)
    averagePathLengthList.append(averagePathLength)
    averageDiameter.append(averageDiameter)
    averageClusteringCoefficientList.append(averageClusteringCoefficient)



# Plot 1 - Network parameters vs error
fileName = "Attachment_Vs_Error"
scatter = plot.ScatterPlot(folderName+fileName, "Scale Free Networks Optimization using GA", "Attachment vs Performance", "Attachment", "Mean Square Error")
scatter.setSeries("Network Performance", np.array(attachmentList), np.array(errorList))
scatter.createOutput()


# Plot 2 - Average degree vs error
fileName = "AverageDegree_Vs_Error"
scatter = plot.ScatterPlot(folderName+fileName, "Scale Free Networks Optimization using GA", "Average Degree vs Performance", "Average Degree", "Mean Square Error")
scatter.setSeries("Network Performance", np.array(averageDegreeList), np.array(errorList))
scatter.createOutput()



# Plot 3 - Average path length vs error
fileName = "AverageDegree_Vs_Error"
scatter = plot.ScatterPlot(folderName+fileName, "Scale Free Networks Optimization using GA", "Average Degree vs Performance", "Average Degree", "Mean Square Error")
scatter.setSeries("Network Performance", np.array(averageDegreeList), np.array(errorList))
scatter.createOutput()

# Plot 4 - Average diameter vs error
fileName = "AverageDegree_Vs_Error"
scatter = plot.ScatterPlot(folderName+fileName, "Scale Free Networks Optimization using GA", "Average Degree vs Performance", "Average Degree", "Mean Square Error")
scatter.setSeries("Network Performance", np.array(averageDegreeList), np.array(errorList))
scatter.createOutput()

# Plot 5 - Average clustering coefficient vs error
fileName = "AverageDegree_Vs_Error"
scatter = plot.ScatterPlot(folderName+fileName, "Scale Free Networks Optimization using GA", "Average Degree vs Performance", "Average Degree", "Mean Square Error")
scatter.setSeries("Network Performance", np.array(averageDegreeList), np.array(errorList))
scatter.createOutput()
