import numpy as np

def splitData(data, trainingRatio, validationRatio, testingRatio):
    N = data.shape[0]
    nTraining = int(trainingRatio*N)
    nValidation = int(validationRatio*N)
    nTesting = N - (nTraining+nValidation)

    # Training data
    trainingData = data[:nTraining]

    # Validation data
    validationData = data[nTraining:nTraining+nValidation]

    # Testing data
    testingData = data[nTraining+nValidation:]

    return trainingData,validationData, testingData

def splitData2(data, trainingRatio):
    N = data.shape[0]
    nTraining = int(trainingRatio*N)

    # Training data
    trainingData = data[:nTraining]

    # Validation data
    validationData = data[nTraining:]

    return trainingData, validationData

def formFeatureVectors(data):
    size = data.shape[0]
    inputTrainingData = np.hstack((np.ones((size-1, 1)),data[:size-1].reshape((size-1, 1))))
    outputTrainingData = data[1:size].reshape((size-1, 1))
    return inputTrainingData, outputTrainingData

def predictFuture(network, seed, horizon):
    # Predict future values
    predictedTestOutputData = []
    lastAvailableData = seed
    for i in range(horizon):
        query = [1.0]
        query.append(lastAvailableData)

        #Predict the next point
        nextPoint = network.predictOnePoint(np.array(query).reshape((1,2)))[0]
        predictedTestOutputData.append(nextPoint)

        lastAvailableData = nextPoint

    predictedTestOutputData = np.array(predictedTestOutputData).reshape((horizon, 1))
    return predictedTestOutputData

