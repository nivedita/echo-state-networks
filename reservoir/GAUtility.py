from reservoir import ReservoirTopology as topology, GATuner as tuner, classicESN as esn, GARandomGraphTuner as rgTuner
import numpy as np
from enum import Enum
from plotting import ScatterPlot as plot
import pickle

class Topology(Enum):
    Classic = 0
    Random = 1
    ErdosRenyi = 2
    SmallWorldGraphs = 3
    ScaleFreeNetworks = 4

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

def tuneTrainPredictGA(trainingInputData, trainingOutputData, validationOutputData,
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
                                             reservoirWeightMatrix=reservoirWeightMatrix)
    spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum = resTuner.getOptimalParameters()

    #Train
    network = esn.Reservoir(size=size,
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
    return predictedOutputData



def tuneTrainPredictConnectivityGA(trainingInputData, trainingOutputData, validationOutputData,
                                  initialInputSeedForValidation, horizon, noOfBest, size=256,initialTransient=50,
                                  resTopology = Topology.Random,
                                  popSize=100, maxGeneration=100):

    # Other reservoir parameters
    spectralRadius = 0.79
    inputScaling = 0.5
    reservoirScaling = 0.5
    leakingRate = 0.3

    if(resTopology == Topology.Random):
        resTuner = rgTuner.RandomGraphTuner(size=size,
                                            initialTransient=initialTransient,
                                            trainingInputData=trainingInputData,
                                            trainingOutputData=trainingOutputData,
                                            initialSeed=initialInputSeedForValidation,
                                            validationOutputData=validationOutputData,
                                            noOfBest=noOfBest,
                                            spectralRadius=spectralRadius, inputScaling=inputScaling,
                                            reservoirScaling=reservoirScaling, leakingRate=leakingRate,
                                            populationSize=popSize, maxGeneration=maxGeneration)
        resTuner.__tune__()
        reservoirConnectivityOptimum = resTuner.getOptimalParameters()
        bestPopulation = resTuner.getBestPopulation()

        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivityOptimum).generateWeightMatrix()

    elif(resTopology == Topology.ErdosRenyi):
        resTuner = rgTuner.ErdosRenyiTuner(size=size,
                                           initialTransient=initialTransient,
                                           trainingInputData=trainingInputData,
                                           trainingOutputData=trainingOutputData,
                                           initialSeed=initialInputSeedForValidation,
                                           validationOutputData=validationOutputData,
                                           noOfBest=noOfBest,
                                           spectralRadius=spectralRadius, inputScaling=inputScaling,
                                           reservoirScaling=reservoirScaling, leakingRate=leakingRate,
                                           populationSize=popSize, maxGeneration=maxGeneration)
        resTuner.__tune__()
        probabilityOptimum = resTuner.getOptimalParameters()
        bestPopulation = resTuner.getBestPopulation()

        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.ErdosRenyiTopology(size=size, probability=probabilityOptimum).generateWeightMatrix()

    elif(resTopology == Topology.ScaleFreeNetworks):
        resTuner = rgTuner.ScaleFreeNetworksTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingInputData=trainingInputData,
                                                 trainingOutputData=trainingOutputData,
                                                 initialSeed=initialInputSeedForValidation,
                                                 validationOutputData=validationOutputData,
                                                 noOfBest=noOfBest,
                                                 spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                 reservoirScaling=reservoirScaling, leakingRate=leakingRate,
                                                 populationSize=popSize, maxGeneration=maxGeneration)
        resTuner.__tune__()
        attachmentOptimum = resTuner.getOptimalParameters()
        bestPopulation = resTuner.getBestPopulation()

        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.ScaleFreeNetworks(size=size, attachmentCount=attachmentOptimum).generateWeightMatrix()
    elif(resTopology == Topology.SmallWorldGraphs):
        resTuner = rgTuner.SmallWorldNetworksTuner(size=size,
                                                  initialTransient=initialTransient,
                                                  trainingInputData=trainingInputData,
                                                  trainingOutputData=trainingOutputData,
                                                  initialSeed=initialInputSeedForValidation,
                                                  validationOutputData=validationOutputData,
                                                  noOfBest=noOfBest,
                                                  spectralRadius=spectralRadius, inputScaling=inputScaling,
                                                  reservoirScaling=reservoirScaling, leakingRate=leakingRate,
                                                  populationSize=popSize, maxGeneration=maxGeneration)
        resTuner.__tune__()
        meanDegreeOptimum, betaOptimum  = resTuner.getOptimalParameters()
        bestPopulation = resTuner.getBestPopulation()

        inputWeightMatrix = topology.ClassicInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.SmallWorldGraphs(size=size, meanDegree=int(meanDegreeOptimum), beta=betaOptimum).generateWeightMatrix()

    #Train
    network = esn.Reservoir(size=size,
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

    initialInputSeedForTesting = validationOutputData[-1]

    predictedOutputData = predictFuture(network, initialInputSeedForTesting, horizon)[:,0]
    return predictedOutputData, bestPopulation


def plotNetworkPerformance(bestPopulation, topology, fileName, networkSize):

    if(topology == Topology.Random):

        # Connectivity on x-axis and error on y-axis
        xAxis = []
        yAxis = []

        for item in bestPopulation:
            connectivity = item[0][0,0]
            error = item[1]
            xAxis.append(connectivity)
            yAxis.append(error)

        xAxis = np.array(xAxis)
        yAxis = np.array(yAxis)
        seriesName = "Network Size "+str(networkSize)

        # Plot the results
        scatter = plot.ScatterPlot(fileName, "Random Graph Connectivity Optimization using GA", "Connectivity vs Performance", "Connectivity", "MSE")
        scatter.setSeries(seriesName, xAxis, yAxis)
        scatter.createOutput()

def getNetworkStats(bestPopulation, type, size):

    log = []
    for item in bestPopulation:
        network = item[2]
        stats = {}
        stats["averageDegree"] = network.networkStats.getAverageDegree()
        stats["averagePathLength"] = network.networkStats.getAveragePathLenth()
        stats["averageDiameter"] = network.networkStats.getDiameter()
        stats["averageClusteringCoefficient"] = network.networkStats.getAverageClusteringCoefficient()

        log.append((item[0], item[1], stats))
    return log


def storeBestPopulationAndStats(bestPopulation, fileName, topology, size):
    output = open(fileName, 'wb')

    print("Writing network stats...")

    # Before dumping, get the stats
    bestPopulation = getNetworkStats(bestPopulation, topology, size)

    pickle.dump(bestPopulation, output)
    output.close()

def loadBestPopulation(fileName):
    pkl_file = open(fileName, 'rb')
    bestPopulation = pickle.load(pkl_file)
    return bestPopulation