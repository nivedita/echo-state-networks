#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from plotting import OutputPlot as outputPlot
import numpy as np
from reservoir import Utility as util
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
trainingData, validationData, testingData = util.splitData(data, 0.5, 0.25, 0.25)
nValidation = validationData.shape[0]
nTesting = testingData.shape[0]

# Form feature vectors for training data
trainingInputData, trainingOutputData = util.formFeatureVectors(trainingData)
#actualOutputData = minMax.inverse_transform(np.vstack((validationData[:nValidation],testingData[:nTesting])))[:,0]
actualOutputData = minMax.inverse_transform(testingData)[:,0]

# Initial seed
initialSeedForValidation = trainingData[-1]

predictedOutputData, error = util.tuneTrainPredict(trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            validationOutputData=validationData,
                                            initialInputSeedForValidation=initialSeedForValidation,
                                            testingData=actualOutputData,
                                            size=500
                                            )


predictedOutputData = minMax.inverse_transform(predictedOutputData)

#Plotting of the prediction output and error
outputFolderName = "Outputs/Outputs" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(outputFolderName)
outplot = outputPlot.OutputPlot(outputFolderName + "/Prediction.html", "Mackey-Glass Time Series - Differential Evolution Optimization", "Predicted vs Actual", "Time", "Output")
outplot.setXSeries(np.arange(1, nTesting + 1))
outplot.setYSeries('Actual Output', actualOutputData)
outplot.setYSeries('Predicted Output', predictedOutputData)
outplot.createOutput()

endTime = time()
run_time = endTime - startTime
print("The run time:"+str(run_time))
print("Done!")