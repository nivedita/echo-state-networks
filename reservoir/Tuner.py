from scipy import optimize
from reservoir import EchoStateNetwork, ReservoirTopology as topology
from performance import RootMeanSquareError as rmse
import numpy as np
import gc

class ESNTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, inputScalingBound, reservoirScalingBound, leakingRateBound,
                 reservoirTopology, inputConnectivity=0.6):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.inputScalingBound = inputScalingBound
        self.reservoirScalingBound = reservoirScalingBound
        self.leakingRateBound = leakingRateBound
        self.reservoirTopology = reservoirTopology
        self.inputConnectivity = inputConnectivity

        #Create a echo state network
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=self.reservoirTopology,
                                                inputConnectivity=self.inputConnectivity)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

    def __ESNTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]
        inputScaling = x[1]

        reservoirScaling = x[2]
        leakingRate = x[3]

        #Create the reservoir
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=self.reservoirTopology,
                                                spectralRadius=spectralRadius,
                                                inputScaling=inputScaling,
                                                reservoirScaling=reservoirScaling,
                                                leakingRate=leakingRate,
                                                initialTransient=self.initialTransient,
                                                inputWeightConn=self.inputWeightConn,
                                                reservoirWeightConn=self.reservoirWeightConn
                                                )

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error

        print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

class ESNTunerWithInitialGuess:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, inputScalingBound, reservoirScalingBound, leakingRateBound,
                 reservoirTopology, inputConnectivity=0.6, initialGuess=None):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.inputScalingBound = inputScalingBound
        self.reservoirScalingBound = reservoirScalingBound
        self.leakingRateBound = leakingRateBound
        self.reservoirTopology = reservoirTopology
        self.inputConnectivity = inputConnectivity

        if(initialGuess == None):
            self.initialGuess = [0.79, 0.5, 0.5, 0.3]
        else:
            self.initialGuess = initialGuess

        #Create a echo state network
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=self.reservoirTopology,
                                                inputConnectivity=self.inputConnectivity)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

    def __ESNTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]
        inputScaling = x[1]
        reservoirScaling = x[2]
        leakingRate = x[3]

        #Create the reservoir
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=self.reservoirTopology,
                                                spectralRadius=spectralRadius,
                                                inputScaling=inputScaling,
                                                reservoirScaling=reservoirScaling,
                                                leakingRate=leakingRate,
                                                initialTransient=self.initialTransient,
                                                inputWeightConn=self.inputWeightConn,
                                                reservoirWeightConn=self.reservoirWeightConn
                                                )

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error

        print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        #result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        result = optimize.basinhopping(func=self.__ESNTrain__, x0=self.initialGuess, stepsize=0.05, niter=1)
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

class ESNTunerWithConnectivity:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, inputScalingBound, reservoirScalingBound, leakingRateBound, inputConnectivityBound, reservoirConnectivityBound):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.inputScalingBound = inputScalingBound
        self.reservoirScalingBound = reservoirScalingBound
        self.leakingRateBound = leakingRateBound
        self.inputConnectivityBound = inputConnectivityBound
        self.reservoirConnectivityBound = reservoirConnectivityBound

        self.inputWeightConn = None
        self.reservoirWeightConn = None

    def __ESNConnTrain__(self, x):
        #Extract the parameters
        inputConnectivity = x[0]
        reservoirConnectivity = x[1]
        reservoirTopology = topology.RandomTopology(size=self.size, connectivity=reservoirConnectivity)

        #Create the network
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=reservoirTopology,
                                                inputConnectivity=inputConnectivity
                                                )

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error

        print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __ESNTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]
        inputScaling = x[1]
        reservoirScaling = x[2]
        leakingRate = x[3]

        #Create the reservoir
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                spectralRadius=spectralRadius,
                                                inputScaling=inputScaling,
                                                reservoirScaling=reservoirScaling,
                                                leakingRate=leakingRate,
                                                initialTransient=self.initialTransient,
                                                inputWeightConn=self.inputWeightConn,
                                                reservoirWeightConn=self.reservoirWeightConn,
                                                reservoirTopology=topology.RandomTopology(self.size, self.reservoirConnectivityOptimum))

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error

        print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):

        # First tune for the input connectivity and the reservoir connectivity
        connBounds = [self.inputConnectivityBound,self.reservoirConnectivityBound]
        connResult = optimize.differential_evolution(self.__ESNConnTrain__,bounds=connBounds)
        self.inputConnectivityOptimum = connResult.x[0]
        self.reservoirConnectivityOptimum = connResult.x[1]

        # With tuned parameters, create the network with optimal connections and keep the connections as same
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=topology.RandomTopology(size=self.size,
                                                                                          connectivity=self.reservoirConnectivityOptimum),
                                                inputConnectivity=self.inputConnectivityOptimum)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

        # Tune the other parameters
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

class ESNMinimalTunerWithConnectivity:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, leakingRateBound, inputConnectivityBound, reservoirConnectivityBound):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.leakingRateBound = leakingRateBound
        self.inputConnectivityBound = inputConnectivityBound
        self.reservoirConnectivityBound = reservoirConnectivityBound
        self.inputWeightConn = None
        self.reservoirWeightConn = None

    def __ESNConnTrain__(self, x):
        #Extract the parameters
        inputConnectivity = x[0]
        reservoirConnectivity = x[1]
        reservoirTopology = topology.RandomTopology(size=self.size, connectivity=reservoirConnectivity)

        #Create the network
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=reservoirTopology,
                                                inputConnectivity=inputConnectivity
                                                )

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error
        print("Optimizing connectivity..")
        print("\nRegression error:"+str(regressionError)+"\n")
        return regressionError

    def __ESNTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]
        leakingRate = x[1]

        #Create the reservoir
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                spectralRadius=spectralRadius,
                                                leakingRate=leakingRate,
                                                initialTransient=self.initialTransient,
                                                inputWeightConn=self.inputWeightConn,
                                                reservoirWeightConn=self.reservoirWeightConn,
                                                reservoirTopology=topology.RandomTopology(self.size, self.reservoirConnectivityOptimum))

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error
        print("Optimizing spectral radius and leaking rate...")
        print("\nRegression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):

        # First tune for the input connectivity and the reservoir connectivity
        connBounds = [self.inputConnectivityBound,self.reservoirConnectivityBound]
        connResult = optimize.differential_evolution(self.__ESNConnTrain__,bounds=connBounds, maxiter=1)
        self.inputConnectivityOptimum = connResult.x[0]
        self.reservoirConnectivityOptimum = connResult.x[1]

        # With tuned parameters, create the network with optimal connections and keep the connections as same
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=topology.RandomTopology(size=self.size,
                                                                                          connectivity=self.reservoirConnectivityOptimum),
                                                inputConnectivity=self.inputConnectivityOptimum)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

        # Tune the other parameters
        bounds = [self.spectralRadiusBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds, maxiter=1)
        return result.x[0], result.x[1], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

class ESNMinimalTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, reservoirTopology, inputConnectivity=0.6):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.reservoirTopology = reservoirTopology
        self.inputConnectivity = inputConnectivity

        #Create a echo state network
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=self.reservoirTopology,
                                                inputConnectivity=self.inputConnectivity)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

    def __ESNTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]

        #Create the reservoir
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=self.reservoirTopology,
                                                spectralRadius=spectralRadius,
                                                initialTransient=self.initialTransient,
                                                inputWeightConn=self.inputWeightConn,
                                                reservoirWeightConn=self.reservoirWeightConn
                                                )

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error

        print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.spectralRadiusBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

class ESNConnTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 inputConnectivityBound, reservoirConnectivityBound):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.inputConnectivityBound = inputConnectivityBound
        self.reservoirConnectivityBound = reservoirConnectivityBound


    def __ESNTrain__(self, x):

        #Extract the parameters
        inputConnectivity = x[0]
        reservoirConnectivity = x[1]
        reservoirTopology = topology.RandomTopology(size=self.size, connectivity=reservoirConnectivity)

        #Create the network
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=reservoirTopology,
                                                inputConnectivity=inputConnectivity
                                                )

        #Train the reservoir
        esn.trainReservoir()

        #Predict for the validation data
        predictedOutputData = esn.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Free the memory
        gc.collect()

        #Return the error
        print("Optimizing connectivity..")
        print("\nRegression error:"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.inputConnectivityBound, self.reservoirConnectivityBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], result.x[1]

        # Brute Force

    def getOptimalParameters(self):
        return self.__tune__()