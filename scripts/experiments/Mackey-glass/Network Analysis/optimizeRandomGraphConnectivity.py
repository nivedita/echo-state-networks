#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from plotting import OutputPlot as outputPlot
import numpy as np
from reservoir import Utility as util, GAUtility as utilityGA
from sklearn import preprocessing as pp
import os
from datetime import datetime
from timeit import default_timer as time

startTime = time()

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data)

#Get only 6000 points
data = data[:6000].reshape((6000, 1))

# Split the data into training, validation and testing
trainingData, validationData, testingData = util.splitData(data, 0.6, 0.3, 0.1)
nValidation = validationData.shape[0]
nTesting = testingData.shape[0]

# Form feature vectors for training data
trainingInputData, trainingOutputData = util.formFeatureVectors(trainingData)
actualOutputData = minMax.inverse_transform(testingData)[:,0]

# Initial seed
initialSeedForValidation = trainingData[-1]
networkSize = 500
populationSize = 10
noOfBest = int(populationSize/2)
noOfGenerations = 10
predictedOutputData, bestPopulation = utilityGA.tuneTrainPredictConnectivityGA(trainingInputData=trainingInputData,
                                                                               trainingOutputData=trainingOutputData,
                                                                               validationOutputData=validationData,
                                                                               initialInputSeedForValidation=initialSeedForValidation,
                                                                               horizon=nTesting,
                                                                               noOfBest=noOfBest,
                                                                               resTopology=utilityGA.Topology.Random,
                                                                               size=networkSize,
                                                                               popSize=populationSize,
                                                                               maxGeneration=noOfGenerations)

predictedOutputData = minMax.inverse_transform(predictedOutputData)

#Plotting of the prediction output and error
outputFolderName = "Outputs/Random_Graph_Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series - GA Optimization)", "Prediction on Testing data", "Time", "Output")
outplot.setXSeries(np.arange(1, nValidation + nTesting + 1))
outplot.setYSeries('Actual Output', actualOutputData)
outplot.setYSeries('Predicted Output', predictedOutputData)
outplot.createOutput()

# Plotting of the best population details
utilityGA.plotNetworkPerformance(bestPopulation, topology=utilityGA.Topology.Random, fileName=outputFolderName+"/NetworkPerformance.html", networkSize=networkSize)

# Store the best population in a file (for later analysis)
popFileName = outputFolderName+"/population.pkl"
utilityGA.storeBestPopulationAndStats(bestPopulation, popFileName, utilityGA.Topology.Random, networkSize)

# Load the best population
print(utilityGA.loadBestPopulation(popFileName))

endTime = time()
run_time = endTime - startTime
print("The run time:"+str(run_time))
print("Done!")