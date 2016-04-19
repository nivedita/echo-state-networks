import numpy as np
from GAOptimizer import Optimizer as opt, MutationOperators as mOperators, CrossoverOperators as cOperators
from reservoir import classicESN as esn, Utility as util, ReservoirTopology as topology
from performance import ErrorMetrics as metrics
import gc


class RandomNetworkErrorObjective():

    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 validationOutputData, initialSeed, horizon,
                 spectralRadius, inputScaling, reservoirScaling, leakingRate):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationOutputData = validationOutputData
        self.initialSeed = initialSeed
        self.horizon = horizon
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate
        self.inputN, self.inputD = self.trainingInputData.shape

    def evaluate(self, x):

        # Extract the parameters
        reservoirConnectivity = x[0,0]

        # To get rid off the randomness in assigning weights, run it 10 times and take the average error
        times = 1
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            network = topology.RandomReservoirTopology(size=self.size, connectivity=reservoirConnectivity)
            reservoirWeightMatrix = network.generateWeightMatrix()

            # Create the reservoir
            res = esn.Reservoir(size=self.size,
                                spectralRadius=self.spectralRadius,
                                inputScaling=self.inputScaling,
                                reservoirScaling=self.reservoirScaling,
                                leakingRate=self.leakingRate,
                                initialTransient=self.initialTransient,
                                inputData=self.trainingInputData,
                                outputData=self.trainingOutputData,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix)

            # Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            # Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            # Calculate the regression error
            errorFunction = metrics.MeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        # Return the error
        #print("Connectivity: "+str(reservoirConnectivity) + "Error: "+str(regressionError))
        return regressionError, network


class RandomGraphTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, noOfBest,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3,
                 populationSize=100, maxGeneration=100):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.noOfBest = noOfBest
        self.horizon = self.validationOutputData.shape[0]
        self.connectivityBound = (0.01,1.0)
        self.populationSize = populationSize
        self.maxGeneration = maxGeneration

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

    def __tune__(self):
        # Create the GA Optimizer
        objective = RandomNetworkErrorObjective(size=self.size, initialTransient=self.initialTransient,
                                                trainingInputData=self.trainingInputData,
                                                trainingOutputData=self.trainingOutputData,
                                                validationOutputData=self.validationOutputData,
                                                initialSeed=self.initialSeed,
                                                horizon=self.horizon,
                                                spectralRadius=self.spectralRadius,
                                                inputScaling=self.inputScaling,
                                                reservoirScaling=self.reservoirScaling,
                                                leakingRate=self.leakingRate)

        self.optimizer = opt.Optimizer(populationSize=self.populationSize,
                                       parameterBounds=[self.connectivityBound],
                                       fitnessObj=objective,
                                       crossoverOperator=cOperators.LineCrossover(),
                                       mutationOperator=mOperators.SimpleAndUniformHybrid(),
                                       maxGeneration=self.maxGeneration,
                                       mutationRate=0.01,
                                       elitismRate=0.5)

        self.optimizer.optimize()

    def getOptimalParameters(self):
        # Return the optimal parameters
        result = self.optimizer.getOptimalParameters()[0][0].tolist()
        connectivityOptimal = result[0]
        return connectivityOptimal

    def getBestPopulation(self):

        # Return the best population and their error values
        population = self.optimizer.getBestParameters(self.noOfBest)
        return population


class ErdosRenyiNetworkErrorObjective():

    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 validationOutputData, initialSeed, horizon,
                 spectralRadius, inputScaling, reservoirScaling, leakingRate):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationOutputData = validationOutputData
        self.initialSeed = initialSeed
        self.horizon = horizon
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate
        self.inputN, self.inputD = self.trainingInputData.shape

    def evaluate(self, x):

        # Extract the parameters
        probability = x[0,0]

        # To get rid off the randomness in assigning weights, run it 10 times and take the average error
        times = 1
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            network = topology.ErdosRenyiTopology(size=self.size, probability=probability)
            reservoirWeightMatrix = network.generateWeightMatrix()

            # Create the reservoir
            res = esn.Reservoir(size=self.size,
                                spectralRadius=self.spectralRadius,
                                inputScaling=self.inputScaling,
                                reservoirScaling=self.reservoirScaling,
                                leakingRate=self.leakingRate,
                                initialTransient=self.initialTransient,
                                inputData=self.trainingInputData,
                                outputData=self.trainingOutputData,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix)

            # Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            # Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            # Calculate the regression error
            errorFunction = metrics.MeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        # Return the error
        #print("Probability: "+str(probability) + "Error: "+str(regressionError))
        return regressionError, network

class ErdosRenyiTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, noOfBest,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3,
                 populationSize=100, maxGeneration=100):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.noOfBest = noOfBest
        self.horizon = self.validationOutputData.shape[0]
        self.probabilityBound = (0.01,1.0)
        self.populationSize = populationSize
        self.maxGeneration = maxGeneration

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

    def __tune__(self):
        # Create the GA Optimizer
        objective = ErdosRenyiNetworkErrorObjective(size=self.size, initialTransient=self.initialTransient,
                                                trainingInputData=self.trainingInputData,
                                                trainingOutputData=self.trainingOutputData,
                                                validationOutputData=self.validationOutputData,
                                                initialSeed=self.initialSeed,
                                                horizon=self.horizon,
                                                spectralRadius=self.spectralRadius,
                                                inputScaling=self.inputScaling,
                                                reservoirScaling=self.reservoirScaling,
                                                leakingRate=self.leakingRate)

        self.optimizer = opt.Optimizer(populationSize=self.populationSize,
                                       parameterBounds=[self.probabilityBound],
                                       fitnessObj=objective,
                                       crossoverOperator=cOperators.LineCrossover(),
                                       mutationOperator=mOperators.SimpleAndUniformHybrid(),
                                       maxGeneration=self.maxGeneration,
                                       mutationRate=0.01,
                                       elitismRate=0.5)

        self.optimizer.optimize()

    def getOptimalParameters(self):
        # Return the optimal parameters
        result = self.optimizer.getOptimalParameters()[0][0].tolist()
        probabilityOptimal = result[0]
        return probabilityOptimal

    def getBestPopulation(self):

        # Return the best population and their error values
        population = self.optimizer.getBestParameters(self.noOfBest)
        return population





class ScaleFreeNetworkErrorObjective():

    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 validationOutputData, initialSeed, horizon,
                 spectralRadius, inputScaling, reservoirScaling, leakingRate):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationOutputData = validationOutputData
        self.initialSeed = initialSeed
        self.horizon = horizon
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate
        self.inputN, self.inputD = self.trainingInputData.shape

    def evaluate(self, x):

        # Extract the parameters
        attachment = int(x[0,0])

        # To get rid off the randomness in assigning weights, run it 10 times and take the average error
        times = 1
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            network = topology.ScaleFreeNetworks(size=self.size, attachmentCount=attachment)
            reservoirWeightMatrix = network.generateWeightMatrix()

            # Create the reservoir
            res = esn.Reservoir(size=self.size,
                                spectralRadius=self.spectralRadius,
                                inputScaling=self.inputScaling,
                                reservoirScaling=self.reservoirScaling,
                                leakingRate=self.leakingRate,
                                initialTransient=self.initialTransient,
                                inputData=self.trainingInputData,
                                outputData=self.trainingOutputData,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix)

            # Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            # Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            # Calculate the regression error
            errorFunction = metrics.MeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        # Return the error
        #print("Attachment: "+str(attachment) + "Error: "+str(regressionError))
        return regressionError, network

class ScaleFreeNetworksTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, noOfBest,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3,
                 populationSize=100, maxGeneration=100):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.noOfBest = noOfBest
        self.horizon = self.validationOutputData.shape[0]
        self.attachmentBound = (1,size-1)
        self.populationSize = populationSize
        self.maxGeneration = maxGeneration

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

    def __tune__(self):
        # Create the GA Optimizer
        objective = ScaleFreeNetworkErrorObjective(size=self.size, initialTransient=self.initialTransient,
                                                   trainingInputData=self.trainingInputData,
                                                   trainingOutputData=self.trainingOutputData,
                                                   validationOutputData=self.validationOutputData,
                                                   initialSeed=self.initialSeed,
                                                   horizon=self.horizon,
                                                   spectralRadius=self.spectralRadius,
                                                   inputScaling=self.inputScaling,
                                                   reservoirScaling=self.reservoirScaling,
                                                   leakingRate=self.leakingRate)

        self.optimizer = opt.Optimizer(populationSize=self.populationSize,
                                       parameterBounds=[self.attachmentBound],
                                       fitnessObj=objective,
                                       crossoverOperator=cOperators.LineCrossover(),
                                       mutationOperator=mOperators.SimpleAndUniformHybrid(),
                                       maxGeneration=self.maxGeneration,
                                       mutationRate=0.01,
                                       elitismRate=0.5)

        self.optimizer.optimize()

    def getOptimalParameters(self):
        # Return the optimal parameters
        result = self.optimizer.getOptimalParameters()[0][0].tolist()
        attachmentOptimal = int(result[0])
        return attachmentOptimal

    def getBestPopulation(self):

        # Return the best population and their error values
        population = self.optimizer.getBestParameters(self.noOfBest)
        return population


class SmaleWorldGraphsErrorObjective():

    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 validationOutputData, initialSeed, horizon,
                 spectralRadius, inputScaling, reservoirScaling, leakingRate):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationOutputData = validationOutputData
        self.initialSeed = initialSeed
        self.horizon = horizon
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate
        self.inputN, self.inputD = self.trainingInputData.shape

    def evaluate(self, x):

        # Extract the parameters
        meanDegree = int(x[0,0])
        beta = x[0,1]

        # To get rid off the randomness in assigning weights, run it 10 times and take the average error
        times = 1
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            network = topology.SmallWorldGraphs(size=self.size, meanDegree=meanDegree, beta=beta)
            reservoirWeightMatrix = network.generateWeightMatrix()

            # Create the reservoir
            res = esn.Reservoir(size=self.size,
                                spectralRadius=self.spectralRadius,
                                inputScaling=self.inputScaling,
                                reservoirScaling=self.reservoirScaling,
                                leakingRate=self.leakingRate,
                                initialTransient=self.initialTransient,
                                inputData=self.trainingInputData,
                                outputData=self.trainingOutputData,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix)

            # Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            # Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            # Calculate the regression error
            errorFunction = metrics.MeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        # Return the error
        #print("SMG parameters: "+str(x) + "Error: "+str(regressionError))
        return regressionError, network

class SmallWorldNetworksTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, noOfBest,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3,
                 populationSize=100, maxGeneration=100):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.noOfBest = noOfBest
        self.horizon = self.validationOutputData.shape[0]
        self.parameterBound = [(2,size-1),(0.01,1.0)]
        self.populationSize = populationSize
        self.maxGeneration = maxGeneration

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

    def __tune__(self):
        # Create the GA Optimizer
        objective = SmaleWorldGraphsErrorObjective(size=self.size, initialTransient=self.initialTransient,
                                                   trainingInputData=self.trainingInputData,
                                                   trainingOutputData=self.trainingOutputData,
                                                   validationOutputData=self.validationOutputData,
                                                   initialSeed=self.initialSeed,
                                                   horizon=self.horizon,
                                                   spectralRadius=self.spectralRadius,
                                                   inputScaling=self.inputScaling,
                                                   reservoirScaling=self.reservoirScaling,
                                                   leakingRate=self.leakingRate)

        self.optimizer = opt.Optimizer(populationSize=self.populationSize,
                                       parameterBounds=self.parameterBound,
                                       fitnessObj=objective,
                                       crossoverOperator=cOperators.SinglePointAndLineHybridCrossover(),
                                       mutationOperator=mOperators.SimpleAndUniformHybrid(),
                                       maxGeneration=self.maxGeneration,
                                       mutationRate=0.01,
                                       elitismRate=0.5)

        self.optimizer.optimize()

    def getOptimalParameters(self):
        # Return the optimal parameters
        result = self.optimizer.getOptimalParameters()[0][0].tolist()
        meanDegreeOptimal = int(result[0])
        betaOptimal = int(result[1])
        return meanDegreeOptimal, betaOptimal

    def getBestPopulation(self):

        # Return the best population and their error values
        population = self.optimizer.getBestParameters(self.noOfBest)
        return population
