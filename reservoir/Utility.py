import numpy as np
from reservoir import EnhancedClassicTuner as tuner, ReservoirTopology as topology, ClassicESN as ESN, Utility as util

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
    input = np.hstack((np.ones((size-1, 1)),data[:size-1].reshape((size-1, 1))))
    output = data[1:size].reshape((size-1, 1))
    return input, output

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

def tuneTrainPredict(trainingInputData, trainingOutputData, validationOutputData,
                 initialInputSeedForValidation, horizon, size = 256,initialTransient=50,
                 spectralRadiusBound=(0.5,1.0),
                 inputScalingBound=(0.0,1.0),
                 reservoirScalingBound=(0.0,1.0),
                 leakingRateBound=(0.1,0.5),
                 reservoirTopology=None):

    # Generate the input and reservoir weight matrices based on the reservoir topology
    inputWeightMatrix = topology.ClassicInputTopology(size, trainingInputData.shape[1]).generateWeightMatrix()
    if reservoirTopology is None:
        reservoirWeightMatrix = topology.ClassicReservoirTopology(size).generateWeightMatrix()
    else: #TODO - think about matrix multiplication
        reservoirWeightMatrix = reservoirTopology.generateConnectivityMatrix()

    resTuner = tuner.ReservoirTuner(size=size,
                                initialTransient=initialTransient,
                                trainingInputData=trainingInputData,
                                trainingOutputData=trainingOutputData,
                                initialSeed=initialInputSeedForValidation,
                                validationOutputData=validationOutputData,
                                spectralRadiusBound=spectralRadiusBound,
                                inputScalingBound=inputScalingBound,
                                reservoirScalingBound=reservoirScalingBound,
                                leakingRateBound=leakingRateBound,
                                inputWeightMatrix=inputWeightMatrix,
                                reservoirWeightMatrix=reservoirWeightMatrix,
                                minimizer=tuner.Minimizer.DifferentialEvolution)
    spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum = resTuner.getOptimalParameters()

    #Train
    network = ESN.Reservoir(size=size,
                              spectralRadius=spectralRadiusOptimum,
                              inputScaling=inputScalingOptimum,
                              reservoirScaling=reservoirScalingOptimum,
                              leakingRate=leakingRateOptimum,
                              initialTransient=initialTransient,
                              inputData=trainingInputData,
                              outputData=trainingOutputData,
                              inputWeightRandom=inputWeightMatrix,
                              reservoirWeightRandom=reservoirWeightMatrix)
    network.trainReservoir()

    #Warm up
    predictedTrainingOutputData = network.predict(trainingInputData)

    # Predict until horizon
    # TODO - throw away the validation output and return the test output only
    horizon = horizon + validationOutputData.shape[0]
    predictedOutputData = predictFuture(network, initialInputSeedForValidation, horizon)[:,0]
    return predictedOutputData