import numpy as np
from reservoir import EnhancedClassicTuner as tuner, ReservoirTopology as topology, classicESN as ESN, Utility as util
from enum import Enum
from performance import ErrorMetrics as metrics

class Topology(Enum):
    Classic = 0
    Random = 1
    ErdosRenyi = 2
    SmallWorldGraphs = 3
    ScaleFreeNetworks = 4

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
        nextPoint = network.predict(np.array(query).reshape((1,2)))[0]
        predictedTestOutputData.append(nextPoint)

        lastAvailableData = nextPoint

    predictedTestOutputData = np.array(predictedTestOutputData).reshape((horizon, 1))
    return predictedTestOutputData

def tuneTrainPredict(trainingInputData, trainingOutputData, validationOutputData,
                 initialInputSeedForValidation, testingData, size = 256,initialTransient=50,
                 spectralRadiusBound=(0.0,1.0),
                 inputScalingBound=(0.0,1.0),
                 reservoirScalingBound=(0.0,1.0),
                 leakingRateBound=(0.0,1.0),
                 reservoirTopology=None):

    # Generate the input and reservoir weight matrices based on the reservoir topology
    inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
    if reservoirTopology is None:
        reservoirWeightMatrix = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()
    else: #TODO - think about matrix multiplication
        reservoirWeightMatrix = reservoirTopology.generateWeightMatrix()

    resTuner = tuner.ReservoirParameterTuner(size=size,
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

    warmupFeatureVectors, warmTargetVectors = formFeatureVectors(validationOutputData)
    predictedWarmup = network.predict(warmupFeatureVectors[-initialTransient:])

    initialInputSeedForTesing = validationOutputData[-1]

    predictedOutputData = predictFuture(network, initialInputSeedForTesing, testingData.shape[0])[:,0]


    cumError = 0
    times = 100
    for i in range(times):
        # Run for many time and get the average regression error
        regressionError = util.trainAndGetError(size=size,
                                                spectralRadius=spectralRadiusOptimum,
                                                inputScaling=inputScalingOptimum,
                                                reservoirScaling=reservoirScalingOptimum,
                                                leakingRate=leakingRateOptimum,
                                                initialTransient=initialTransient,
                                                trainingInputData=trainingInputData,
                                                trainingOutputData=trainingOutputData,
                                                inputWeightMatrix=inputWeightMatrix,
                                                reservoirWeightMatrix=reservoirWeightMatrix,
                                                validationOutputData=validationOutputData,
                                                horizon=testingData.shape[0],
                                                testingActualOutputData=testingData)
        cumError += regressionError

    error = cumError/times
    return predictedOutputData, error

def tuneTrainPredictConnectivity(trainingInputData, trainingOutputData, validationOutputData,
                                            initialInputSeedForValidation, horizon, size=256,initialTransient=50,
                                            resTopology = Topology.Random):

    # Other reservoir parameters
    spectralRadius = 0.79
    inputScaling = 0.5
    reservoirScaling = 0.5
    leakingRate = 0.3

    if(resTopology == Topology.Random):
        resTuner = tuner.RandomConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        reservoirConnectivityOptimum = resTuner.getOptimalParameters()
        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivityOptimum).generateWeightMatrix()

    elif(resTopology == Topology.ErdosRenyi):
        resTuner = tuner.ErdosRenyiConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        probabilityOptimum = resTuner.getOptimalParameters()
        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.ErdosRenyiTopology(size=size, probability=probabilityOptimum).generateWeightMatrix()

    elif(resTopology == Topology.ScaleFreeNetworks):
        resTuner = tuner.ScaleFreeNetworksConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        attachmentOptimum = resTuner.getOptimalParameters()
        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.ScaleFreeNetworks(size=size, attachmentCount=attachmentOptimum).generateWeightMatrix()
    elif(resTopology == Topology.SmallWorldGraphs):
        resTuner = tuner.SmallWorldGraphsConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        meanDegreeOptimum, betaOptimum  = resTuner.getOptimalParameters()
        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.SmallWorldGraphs(size=size, meanDegree=int(meanDegreeOptimum), beta=betaOptimum).generateWeightMatrix()

    # TODO: train 10 times and get the mean prediction and mean error

    #Train
    network = ESN.Reservoir(size=size,
                            spectralRadius=spectralRadius,
                            inputScaling=inputScaling,
                            reservoirScaling=reservoirScaling,
                            leakingRate=leakingRate,
                            initialTransient=initialTransient,
                            inputData=trainingInputData,
                            outputData=trainingOutputData,
                            inputWeightRandom=inputWeightMatrix,
                            reservoirWeightRandom=reservoirWeightMatrix)
    network.trainReservoir()

    warmupFeatureVectors, warmTargetVectors = formFeatureVectors(validationOutputData)
    predictedWarmup = network.predict(warmupFeatureVectors[-initialTransient:])

    initialInputSeedForTesing = validationOutputData[-1]

    predictedOutputData = predictFuture(network, initialInputSeedForTesing, horizon)[:,0]
    return predictedOutputData


def tuneTrainPredictConnectivityNonBrute(trainingInputData, trainingOutputData, validationOutputData,
                                            initialInputSeedForValidation, horizon, size=256,initialTransient=50,
                                            resTopology = Topology.Random):

    # Other reservoir parameters
    spectralRadius = 0.79
    inputScaling = 0.5
    reservoirScaling = 0.5
    leakingRate = 0.3

    if(resTopology == Topology.Random):
        resTuner = tuner.RandomConnectivityTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        reservoirConnectivityOptimum = resTuner.getOptimalParameters()
        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivityOptimum).generateWeightMatrix()

    elif(resTopology == Topology.SmallWorldGraphs):
        resTuner = tuner.SmallWorldGraphsConnectivityNonBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        meanDegreeOptimum, betaOptimum  = resTuner.getOptimalParameters()
        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.SmallWorldGraphs(size=size, meanDegree=int(meanDegreeOptimum), beta=betaOptimum).generateWeightMatrix()

    #Train
    network = ESN.Reservoir(size=size,
                            spectralRadius=spectralRadius,
                            inputScaling=inputScaling,
                            reservoirScaling=reservoirScaling,
                            leakingRate=leakingRate,
                            initialTransient=initialTransient,
                            inputData=trainingInputData,
                            outputData=trainingOutputData,
                            inputWeightRandom=inputWeightMatrix,
                            reservoirWeightRandom=reservoirWeightMatrix)
    network.trainReservoir()

    warmupFeatureVectors, warmTargetVectors = formFeatureVectors(validationOutputData)
    predictedWarmup = network.predict(warmupFeatureVectors[-initialTransient:])

    initialInputSeedForTesing = validationOutputData[-1]

    predictedOutputData = predictFuture(network, initialInputSeedForTesing, horizon)[:,0]
    return predictedOutputData




def tuneConnectivity(trainingInputData, trainingOutputData, validationOutputData,
                    initialInputSeedForValidation, horizon, testingActualOutputData,
                    size=256,initialTransient=50,
                    resTopology = Topology.Classic):

    # Other reservoir parameters
    spectralRadius = 0.79
    inputScaling = 0.5
    reservoirScaling = 0.5
    leakingRate = 0.3

    # Optimal Parameters List
    optimalParameters = {}


    if(resTopology == Topology.Classic):
        # Run 100 times and get the average regression error
        iterations = 1000
        cumulativeError = 0.0
        for i in range(iterations):
            inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
            reservoirWeightMatrix = topology.ClassicReservoirTopology(size=size).generateWeightMatrix()

            error = trainAndGetError(size=size,
                                     spectralRadius=spectralRadius,
                                     inputScaling=inputScaling,
                                     reservoirScaling=reservoirScaling,
                                     leakingRate=leakingRate,
                                     initialTransient=initialTransient,
                                     trainingInputData=trainingInputData,
                                     trainingOutputData=trainingOutputData,
                                     inputWeightMatrix=inputWeightMatrix,
                                     reservoirWeightMatrix=reservoirWeightMatrix,
                                     validationOutputData=validationOutputData,
                                     horizon=horizon,
                                     testingActualOutputData=testingActualOutputData)

            # Calculate the error
            cumulativeError += error

        return cumulativeError/iterations, optimalParameters

    elif(resTopology == Topology.Random):
        resTuner = tuner.RandomConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        reservoirConnectivityOptimum = resTuner.getOptimalParameters()


        optimalParameters["Optimal_Reservoir_Connectivity"] = reservoirConnectivityOptimum

        # Run 100 times and get the average regression error
        iterations = 1000
        cumulativeError = 0.0
        for i in range(iterations):
            inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
            reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivityOptimum).generateWeightMatrix()

            error = trainAndGetError(size=size,
                                     spectralRadius=spectralRadius,
                                     inputScaling=inputScaling,
                                     reservoirScaling=reservoirScaling,
                                     leakingRate=leakingRate,
                                     initialTransient=initialTransient,
                                     trainingInputData=trainingInputData,
                                     trainingOutputData=trainingOutputData,
                                     inputWeightMatrix=inputWeightMatrix,
                                     reservoirWeightMatrix=reservoirWeightMatrix,
                                     validationOutputData=validationOutputData,
                                     horizon=horizon,
                                     testingActualOutputData=testingActualOutputData)

            # Calculate the error
            cumulativeError += error

        return cumulativeError/iterations, optimalParameters

    elif(resTopology == Topology.ErdosRenyi):
        resTuner = tuner.ErdosRenyiConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        probabilityOptimum = resTuner.getOptimalParameters()

        optimalParameters["Optimal_Connectivity_Probability"] = probabilityOptimum

        # Run 100 times and get the average regression error
        iterations = 1000
        cumulativeError = 0.0
        for i in range(iterations):
            inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
            reservoirWeightMatrix = topology.ErdosRenyiTopology(size=size, probability=probabilityOptimum).generateWeightMatrix()

            error = trainAndGetError(size=size,
                                     spectralRadius=spectralRadius,
                                     inputScaling=inputScaling,
                                     reservoirScaling=reservoirScaling,
                                     leakingRate=leakingRate,
                                     initialTransient=initialTransient,
                                     trainingInputData=trainingInputData,
                                     trainingOutputData=trainingOutputData,
                                     inputWeightMatrix=inputWeightMatrix,
                                     reservoirWeightMatrix=reservoirWeightMatrix,
                                     validationOutputData=validationOutputData,
                                     horizon=horizon,
                                     testingActualOutputData=testingActualOutputData)

            # Calculate the error
            cumulativeError += error

        return cumulativeError/iterations, optimalParameters

    elif(resTopology == Topology.ScaleFreeNetworks):
        resTuner = tuner.ScaleFreeNetworksConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        attachmentOptimum = resTuner.getOptimalParameters()

        optimalParameters["Optimal_Preferential_Attachment"] = attachmentOptimum

        # Run 100 times and get the average regression error
        iterations = 1000
        cumulativeError = 0.0
        for i in range(iterations):
            inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
            reservoirWeightMatrix = topology.ScaleFreeNetworks(size=size, attachmentCount=attachmentOptimum).generateWeightMatrix()

            error = trainAndGetError(size=size,
                                     spectralRadius=spectralRadius,
                                     inputScaling=inputScaling,
                                     reservoirScaling=reservoirScaling,
                                     leakingRate=leakingRate,
                                     initialTransient=initialTransient,
                                     trainingInputData=trainingInputData,
                                     trainingOutputData=trainingOutputData,
                                     inputWeightMatrix=inputWeightMatrix,
                                     reservoirWeightMatrix=reservoirWeightMatrix,
                                     validationOutputData=validationOutputData,
                                     horizon=horizon,
                                     testingActualOutputData=testingActualOutputData)

            # Calculate the error
            cumulativeError += error

        return cumulativeError/iterations, optimalParameters

    elif(resTopology == Topology.SmallWorldGraphs):
        resTuner = tuner.SmallWorldGraphsConnectivityBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate)
        meanDegreeOptimum, betaOptimum  = resTuner.getOptimalParameters()

        optimalParameters["Optimal_MeanDegree"] = meanDegreeOptimum
        optimalParameters["Optimal_Beta"] = betaOptimum

        # Run 100 times and get the average regression error
        iterations = 1000
        cumulativeError = 0.0
        for i in range(iterations):
            inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
            reservoirWeightMatrix = topology.SmallWorldGraphs(size=size, meanDegree=int(meanDegreeOptimum), beta=betaOptimum).generateWeightMatrix()

            error = trainAndGetError(size=size,
                                     spectralRadius=spectralRadius,
                                     inputScaling=inputScaling,
                                     reservoirScaling=reservoirScaling,
                                     leakingRate=leakingRate,
                                     initialTransient=initialTransient,
                                     trainingInputData=trainingInputData,
                                     trainingOutputData=trainingOutputData,
                                     inputWeightMatrix=inputWeightMatrix,
                                     reservoirWeightMatrix=reservoirWeightMatrix,
                                     validationOutputData=validationOutputData,
                                     horizon=horizon,
                                     testingActualOutputData=testingActualOutputData)

            # Calculate the error
            cumulativeError += error

        return cumulativeError/iterations, optimalParameters

def trainAndGetError(size, spectralRadius, inputScaling, reservoirScaling, leakingRate,
                     initialTransient, trainingInputData, trainingOutputData,
                     inputWeightMatrix, reservoirWeightMatrix,
                     validationOutputData, horizon, testingActualOutputData):

    # Error function
    errorFun = metrics.MeanSquareError()

    # Train
    network = ESN.Reservoir(size=size,
                            spectralRadius=spectralRadius,
                            inputScaling=inputScaling,
                            reservoirScaling=reservoirScaling,
                            leakingRate=leakingRate,
                            initialTransient=initialTransient,
                            inputData=trainingInputData,
                            outputData=trainingOutputData,
                            inputWeightRandom=inputWeightMatrix,
                            reservoirWeightRandom=reservoirWeightMatrix)
    network.trainReservoir()

    warmupFeatureVectors, warmTargetVectors = formFeatureVectors(validationOutputData)
    predictedWarmup = network.predict(warmupFeatureVectors[-initialTransient:])

    initialInputSeedForTesing = validationOutputData[-1]

    predictedOutputData = predictFuture(network, initialInputSeedForTesing, horizon)[:,0]

    # Calculate the error
    error = errorFun.compute(testingActualOutputData, predictedOutputData)
    return error