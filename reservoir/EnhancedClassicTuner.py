from scipy import optimize
from reservoir import classicESN, ReservoirTopology as topology
from performance import ErrorMetrics as metrics
import numpy as np
from reservoir import Utility as util
from enum import Enum
import gc
import decimal

class Minimizer(Enum):
    BasinHopping = 1
    DifferentialEvolution = 2
    BruteForce = 3

class ParameterStep(object):
    def __init__(self, stepsize=0.005):
        self.stepsize = stepsize
    def __call__(self, x):
        x += self.stepsize
        return x

class ReservoirParameterTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, spectralRadiusBound, inputScalingBound,
                 reservoirScalingBound, leakingRateBound,inputWeightMatrix=None,
                 reservoirWeightMatrix=None, minimizer=Minimizer.DifferentialEvolution,
                 initialGuess = np.array([0.79, 0.5, 0.5, 0.3])):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.inputScalingBound = inputScalingBound
        self.reservoirScalingBound = reservoirScalingBound
        self.leakingRateBound = leakingRateBound
        self.horizon = self.validationOutputData.shape[0]
        self.minimizer = minimizer
        self.initialGuess = initialGuess

        if inputWeightMatrix is None:
            self.inputN, self.inputD = self.trainingInputData.shape
            self.inputWeightRandom = np.random.rand(self.size, self.inputD)
        else:
            self.inputWeightRandom = inputWeightMatrix

        if reservoirWeightMatrix is None:
            self.reservoirWeightRandom = np.random.rand(self.size, self.size)
        else:
            self.reservoirWeightRandom = reservoirWeightMatrix


    def __reservoirTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]
        inputScaling = x[1]
        reservoirScaling = x[2]
        leakingRate = x[3]

        #Create the reservoir
        res = classicESN.Reservoir(size=self.size,
                                  spectralRadius=spectralRadius,
                                  inputScaling=inputScaling,
                                  reservoirScaling=reservoirScaling,
                                  leakingRate=leakingRate,
                                  initialTransient=self.initialTransient,
                                  inputData=self.trainingInputData,
                                  outputData=self.trainingOutputData,
                                  inputWeightRandom=self.inputWeightRandom,
                                  reservoirWeightRandom=self.reservoirWeightRandom)

        #Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        #Predict for the validation data
        predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

        gc.collect()

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def __tune__(self):
        if self.minimizer == Minimizer.DifferentialEvolution:
            bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
            result = optimize.differential_evolution(self.__reservoirTrain__,bounds=bounds)
            print("The Optimization results are :"+str(result))
            return result.x[0], result.x[1], result.x[2], result.x[3]
        else:
            bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
            minimizer_kwargs = {"method": "TNC", "bounds":bounds, "options": {"eps":0.005}}
            mytakestep = ParameterStep()
            result = optimize.basinhopping(self.__reservoirTrain__, x0=self.initialGuess, minimizer_kwargs=minimizer_kwargs, take_step=mytakestep, stepsize=0.005)
            print("The Optimization results are :"+str(result))
            return result.x[0], result.x[1], result.x[2], result.x[3]

    def getOptimalParameters(self):
        return self.__tune__()

class RandomConnectivityTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, reservoirConnectivityBound,
                 minimizer=Minimizer.DifferentialEvolution, initialGuess=0.5,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.reservoirConnectivityBound = reservoirConnectivityBound
        self.horizon = self.validationOutputData.shape[0]
        self.minimizer = minimizer
        self.initialGuess = np.array([initialGuess])

        # Input-to-reservoir is of Classic Type - Fully connected and maintained as constant
        self.inputN, self.inputD = self.trainingInputData.shape
        self.inputWeight = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

    def __reservoirTrain__(self, x):

        #Extract the parameters
        reservoirConnectivity = x[0]

        # To get rid off the randomness in assigning weights, run it 10 times and  take the average error
        times = 1
        error = 0

        for i in range(times):

            self.inputWeight = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()

            # Reservoir Weight Matrix
            reservoirWeight = topology.RandomReservoirTopology(size=self.size, connectivity=reservoirConnectivity).generateWeightMatrix()

            #Create the reservoir
            res = classicESN.Reservoir(size=self.size,
                                      spectralRadius=self.spectralRadius,
                                      inputScaling=self.inputScaling,
                                      reservoirScaling=self.reservoirScaling,
                                      leakingRate=self.leakingRate,
                                      initialTransient=self.initialTransient,
                                      inputData=self.trainingInputData,
                                      outputData=self.trainingOutputData,
                                      inputWeightRandom=self.inputWeight,
                                      reservoirWeightRandom=reservoirWeight)

            #Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            #Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            #Calcuate the regression error
            errorFunction = rmse.RootMeanSquareError()
            error += errorFunction.compute(self.validationOutputData, predictedOutputData)

        regressionError = error/times

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def __tune__(self):
        if self.minimizer == Minimizer.DifferentialEvolution:
            bounds = [self.reservoirConnectivityBound]
            result = optimize.differential_evolution(self.__reservoirTrain__,bounds=bounds)
            print("The Optimization results are :"+str(result))
            return result.x[0], self.inputWeight
        elif self.minimizer == Minimizer.BasinHopping:
            bounds = [self.reservoirConnectivityBound]
            minimizer_kwargs = {"method": "TNC", "bounds":bounds, "options": {"eps":0.01}}
            mytakestep = ParameterStep()
            result = optimize.basinhopping(self.__reservoirTrain__, x0=self.initialGuess, minimizer_kwargs=minimizer_kwargs, take_step=mytakestep, stepsize=0.01)
            print("The Optimization results are :"+str(result))
            return result.x[0], self.inputWeight
    def getOptimalParameters(self):
        return self.__tune__()

class RandomConnectivityBruteTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.horizon = self.validationOutputData.shape[0]
        self.ranges = slice(0.1,1.0,0.005)

        # Input-to-reservoir is of Classic Type - Fully connected and maintained as constant
        self.inputN, self.inputD = self.trainingInputData.shape

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

    def generateRandomInputWeightMatrix(self):
        return np.random.rand(self.size, self.inputD)

    def generateRandomReservoirWeightMatrix(self):
        return np.random.rand(self.size, self.size)

    def __reservoirTrain__(self, x):

        #Extract the parameters
        reservoirConnectivity = float(x)

        # To get rid off the randomness in assigning weights, run it 10 times and  take the average error
        times = 1
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            reservoirWeightMatrix = topology.RandomReservoirTopology(size=self.size, connectivity=reservoirConnectivity).generateWeightMatrix()

            #Create the reservoir
            res = classicESN.Reservoir(size=self.size,
                                      spectralRadius=self.spectralRadius,
                                      inputScaling=self.inputScaling,
                                      reservoirScaling=self.reservoirScaling,
                                      leakingRate=self.leakingRate,
                                      initialTransient=self.initialTransient,
                                      inputData=self.trainingInputData,
                                      outputData=self.trainingOutputData,
                                      inputWeightRandom=inputWeightMatrix,
                                      reservoirWeightRandom=reservoirWeightMatrix)

            #Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            #Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            #Calcuate the regression error
            errorFunction = metrics.RootMeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def __tune__(self):
        result = optimize.brute(self.__reservoirTrain__,ranges=(self.ranges,), finish=None, full_output=True)
        print("The Optimization results are :"+str(result))
        return result[0]
    def getOptimalParameters(self):
        return self.__tune__()

class ErdosRenyiConnectivityBruteTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.horizon = self.validationOutputData.shape[0]

        # Probability ranges
        self.ranges = slice(0.1,1.0,0.005)

        # Input-to-reservoir is of Classic Type - Fully connected and maintained as constant
        self.inputN, self.inputD = self.trainingInputData.shape

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

        # Dictionary to store the reservoir conn
        self.reservoirConnDict = {}


    def generateRandomInputWeightMatrix(self):
        return np.random.rand(self.size, self.inputD)

    def generateRandomReservoirWeightMatrix(self):
        return np.random.rand(self.size, self.size)

    def __reservoirTrain__(self, x):

        #Extract the parameters
        probability = x

        # To get rid off the randomness in assigning weights, run it 10 times and  take the average error
        times = 10
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            reservoirWeightMatrix = topology.ErdosRenyiTopology(size=self.size, probability=probability).generateWeightMatrix()

            #Create the reservoir
            res = classicESN.Reservoir(size=self.size,
                                      spectralRadius=self.spectralRadius,
                                      inputScaling=self.inputScaling,
                                      reservoirScaling=self.reservoirScaling,
                                      leakingRate=self.leakingRate,
                                      initialTransient=self.initialTransient,
                                      inputData=self.trainingInputData,
                                      outputData=self.trainingOutputData,
                                      inputWeightRandom=inputWeightMatrix,
                                      reservoirWeightRandom=reservoirWeightMatrix)

            #Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            #Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            #Calcuate the regression error
            errorFunction = metrics.RootMeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def __tune__(self):
        result = optimize.brute(self.__reservoirTrain__,ranges=(self.ranges,), finish=None, full_output=True)
        print("The Optimization results are :"+str(result))
        return result[0]
    def getOptimalParameters(self):
        return self.__tune__()

class ScaleFreeNetworksConnectivityBruteTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.horizon = self.validationOutputData.shape[0]

        # Attachment range
        self.ranges = slice(1,self.size - 1,1)

        # Input-to-reservoir is of Classic Type - Fully connected and maintained as constant
        self.inputN, self.inputD = self.trainingInputData.shape

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

        # Dictionary to store the reservoir conn
        self.reservoirConnDict = {}


    def generateRandomInputWeightMatrix(self):
        return np.random.rand(self.size, self.inputD)

    def generateRandomReservoirWeightMatrix(self):
        return np.random.rand(self.size, self.size)

    def __reservoirTrain__(self, x):

        #Extract the parameters
        attachment = int(x)

        # To get rid off the randomness in assigning weights, run it 10 times and  take the average error
        times = 1
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            reservoirWeightMatrix = topology.ScaleFreeNetworks(size=self.size, attachmentCount=attachment).generateWeightMatrix()

            #Create the reservoir
            res = classicESN.Reservoir(size=self.size,
                                      spectralRadius=self.spectralRadius,
                                      inputScaling=self.inputScaling,
                                      reservoirScaling=self.reservoirScaling,
                                      leakingRate=self.leakingRate,
                                      initialTransient=self.initialTransient,
                                      inputData=self.trainingInputData,
                                      outputData=self.trainingOutputData,
                                      inputWeightRandom=inputWeightMatrix,
                                      reservoirWeightRandom=reservoirWeightMatrix)

            #Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            #Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            #Calcuate the regression error
            errorFunction = metrics.RootMeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def __tune__(self):
        result = optimize.brute(self.__reservoirTrain__,ranges=(self.ranges,), finish=None, full_output=True)
        print("The Optimization results are :"+str(result))
        return int(result[0])
    def getOptimalParameters(self):
        return self.__tune__()

class SmallWorldGraphsConnectivityBruteTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData,
                 spectralRadius = 0.79, inputScaling = 0.5, reservoirScaling=0.5, leakingRate=0.3):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeed = initialSeed
        self.validationOutputData = validationOutputData
        self.horizon = self.validationOutputData.shape[0]

        # Ranges for mean degree k and beta
        self.ranges = (slice(2,self.size - 1,2), slice(0.1,1.0,0.01))

        # Input-to-reservoir is of Classic Type - Fully connected and maintained as constant
        self.inputN, self.inputD = self.trainingInputData.shape

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate

        # Dictionary to store the reservoir conn
        self.reservoirConnDict = {}


    def generateRandomInputWeightMatrix(self):
        return np.random.rand(self.size, self.inputD)

    def generateRandomReservoirWeightMatrix(self):
        return np.random.rand(self.size, self.size)

    def __reservoirTrain__(self, x):

        #Extract the parameters
        meanDegree, beta = x
        meanDegree = int(meanDegree)

        # To get rid off the randomness in assigning weights, run it 10 times and  take the average error
        times = 10
        cumulativeError = 0

        for i in range(times):
            # Input and weight connectivity Matrix
            inputWeightMatrix = topology.ClassicInputTopology(self.inputD, self.size).generateWeightMatrix()
            reservoirWeightMatrix = topology.SmallWorldGraphs(size=self.size, meanDegree=meanDegree, beta=beta).generateWeightMatrix()

            #Create the reservoir
            res = classicESN.Reservoir(size=self.size,
                                      spectralRadius=self.spectralRadius,
                                      inputScaling=self.inputScaling,
                                      reservoirScaling=self.reservoirScaling,
                                      leakingRate=self.leakingRate,
                                      initialTransient=self.initialTransient,
                                      inputData=self.trainingInputData,
                                      outputData=self.trainingOutputData,
                                      inputWeightRandom=inputWeightMatrix,
                                      reservoirWeightRandom=reservoirWeightMatrix)

            #Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

            #Predict for the validation data
            predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

            gc.collect()

            #Calcuate the regression error
            errorFunction = metrics.RootMeanSquareError()
            error = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumulativeError += error

        regressionError = cumulativeError/times

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def __tune__(self):
        result = optimize.brute(self.__reservoirTrain__,ranges=self.ranges, finish=None, full_output=True)
        print("The Optimization results are :"+str(result))
        return result[0]
    def getOptimalParameters(self):
        return self.__tune__()
