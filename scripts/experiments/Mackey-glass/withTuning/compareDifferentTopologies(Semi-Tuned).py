#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

import sys
sys.path.append('C:/Users/Raj/PycharmProjects/echo-state-networks')

from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
import numpy as np
from reservoir import Utility as util
from sklearn import preprocessing as pp
import os
from datetime import datetime
from timeit import default_timer as time
from performance import ErrorMetrics as metrics

outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
results = open(outputFolderName+"/OptimalParameters.res", 'w')

startTime = time()

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Normalize the raw data
minMax = pp.MinMaxScaler((-1,1))
data = minMax.fit_transform(data)

#Get only 6000 points
data = data[:6000].reshape((6000, 1))

# Split the data into training, validation and testing
trainingData, validationData, testingData = util.splitData(data, 0.5, 0.25, 0.25)
nValidation = validationData.shape[0]
nTesting = testingData.shape[0]

# Form feature vectors for training data
trainingInputData, trainingOutputData = util.formFeatureVectors(trainingData)
actualOutputData = minMax.inverse_transform(testingData)[:,0]

# Initial seed
initialSeedForValidation = trainingData[-1]

# Error function
error = metrics.MeanSquareError()

# Number of iterations
iterations = 1


# Run Classic ESN
classicESNError = 0
print("\n Running Classic ESN Tuner..")
for i in range(iterations):
    print("\n Iteration:"+str(i+1))
    error, optimalParameters = util.tuneConnectivity(trainingInputData=trainingInputData,
                                                trainingOutputData=trainingOutputData,
                                                validationOutputData=validationData,
                                                initialInputSeedForValidation=initialSeedForValidation,
                                                horizon=nTesting,
                                                testingActualOutputData=testingData,
                                                resTopology=util.Topology.Classic
                                                )

    classicESNError += error

    # Log the results

classicESNError = classicESNError/iterations

# Run random ESN Tuner
randomESNError = 0
print("\n Running Random ESN Tuner..")
results.write("\nRandom ESN Parameters:")
for i in range(iterations):
    print("\n Iteration:"+str(i+1))
    error, optimalParameters = util.tuneConnectivity(trainingInputData=trainingInputData,
                                                trainingOutputData=trainingOutputData,
                                                validationOutputData=validationData,
                                                initialInputSeedForValidation=initialSeedForValidation,
                                                horizon=nTesting,
                                                testingActualOutputData=testingData,
                                                resTopology=util.Topology.Random
                                                )


    # Log the results
    results.write("\nIteration:"+str(i+1)+" Parameters:"+str(optimalParameters))

    randomESNError += error
randomESNError = randomESNError/iterations


# Run Erdos ESN Tuner
erdosESNError = 0
print("\n Running Erdos ESN Tuner..")
results.write("\nErdos ESN Parameters:")
for i in range(iterations):
    print("\n Iteration:"+str(i+1))
    error, optimalParameters = util.tuneConnectivity(trainingInputData=trainingInputData,
                                                trainingOutputData=trainingOutputData,
                                                validationOutputData=validationData,
                                                initialInputSeedForValidation=initialSeedForValidation,
                                                horizon=nTesting,
                                                testingActualOutputData=testingData,
                                                resTopology=util.Topology.ErdosRenyi
                                                )

    # Log the results
    results.write("\nIteration:"+str(i+1)+" Parameters:"+str(optimalParameters))
    erdosESNError += error
erdosESNError = erdosESNError/iterations


# Run Small World Graphs ESN Tuner
smallWorldESNError = 0
print("\n Running Small World Graphs ESN Tuner..")
results.write("\nSmall World Graph ESN Parameters:")
for i in range(iterations):
    print("\n Iteration:"+str(i+1))
    error, optimalParameters = util.tuneConnectivity(trainingInputData=trainingInputData,
                                                trainingOutputData=trainingOutputData,
                                                validationOutputData=validationData,
                                                initialInputSeedForValidation=initialSeedForValidation,
                                                horizon=nTesting,
                                                testingActualOutputData=testingData,
                                                resTopology=util.Topology.SmallWorldGraphs
                                                )


    # Log the results
    results.write("\nIteration:"+str(i+1)+" Parameters:"+str(optimalParameters))

    smallWorldESNError += error
smallWorldESNError = smallWorldESNError/iterations

# Run Scale Free Networks ESN Tuner
scaleFreeESNError = 0
print("\n Running Scale Free Networks ESN Tuner..")
results.write("\nScale Free Networks ESN Parameters:")
for i in range(iterations):
    print("\n Iteration:"+str(i+1))
    error, optimalParameters = util.tuneConnectivity(trainingInputData=trainingInputData,
                                                trainingOutputData=trainingOutputData,
                                                validationOutputData=validationData,
                                                initialInputSeedForValidation=initialSeedForValidation,
                                                horizon=nTesting,
                                                testingActualOutputData=testingData,
                                                resTopology=util.Topology.ScaleFreeNetworks
                                                )

    # Log the results
    results.write("\nIteration:"+str(i+1)+" Parameters:"+str(optimalParameters))

    scaleFreeESNError += error
scaleFreeESNError = scaleFreeESNError/iterations

#Plotting of regression error
errPlot = errorPlot.ErrorPlot(outputFolderName + "/RegressionError.html", "Comparison of different graph topologies", "with connectivity parameters tuned", "ESN Configuration", "Total Error")
errPlot.setXAxis(np.array(['Classic ESN', 'Random ESN', 'Erdos Renyi ESN', 'Small World Graph ESN', 'Scale Free Network ESN']))
errPlot.setYAxis('Mean Squared Error', np.array([classicESNError, randomESNError, erdosESNError, smallWorldESNError, scaleFreeESNError]))
errPlot.createOutput()
