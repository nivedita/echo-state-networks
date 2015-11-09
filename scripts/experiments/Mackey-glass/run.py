
#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import reservoir as reservoir
import numpy as np
import matplotlib.pyplot as plt

#Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

#Input data - get 2000 points
nTraining = 2000
inputData = np.hstack((np.ones((nTraining, 1)),data[:nTraining].reshape((nTraining, 1))))

#Output data
outputData = data[1:nTraining+1].reshape((nTraining, 1))

#Train reservoir
size = 256
spectralRadius = 0.99
inputScaling = 0.5
leakingRate = 0.3
initialTransient = 100
reservoirObj = reservoir.Reservoir(size, spectralRadius, inputScaling, leakingRate, initialTransient, inputData, outputData)
reservoirObj.trainReservoir()

#Test the reservoir
nTesting = 2000
testInputData = inputData = np.hstack((np.ones((nTesting, 1)),data[nTraining:nTraining+nTesting].reshape((nTesting, 1))))
testActualOutputData = data[nTraining+1:nTraining+nTesting+1].reshape((nTraining, 1))
testPredictedOutputData = reservoirObj.predict(testInputData)

#plot real time series data (Utest) and predictions (ytest)
plt.plot(testActualOutputData, 'r')
plt.plot(testPredictedOutputData, 'b')
plt.show()

print ("Done!")

