from scipy import optimize
from reservoir import EchoStateNetwork, ReservoirTopology as topology
from performance import RootMeanSquareError as rmse
import numpy as np
import gc
from enum import Enum

class Minimizer(Enum):
    BasinHopping = 1
    DifferentialEvolution = 2

class ESNParameterBounds(object):
     def __init__(self):
         self.xmax = np.array([1.0, 1.0, 1.0, 1.0])
         self.xmin = np.array([0.0, 0.0, 0.0, 0.0])

     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin

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
        #print("Parameters:"+str(x))
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
        print("\nParameters:"+str(x)+"Regression error"+str(regressionError))
        return regressionError

    def __tune__(self):
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        print("The optimization results:"+str(result))
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

class ESNTunerWithConnectivity:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, inputScalingBound, reservoirScalingBound, leakingRateBound, inputConnectivityBound, reservoirConnectivityBound, times=10):
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
        self.times = times

        self.inputWeightConn = None
        self.reservoirWeightConn = None

    def __ESNConnTrain__(self, x):
        #Extract the parameters
        inputConnectivity = x[0]
        reservoirConnectivity = x[1]
        reservoirTopology = topology.RandomTopology(size=self.size, connectivity=reservoirConnectivity)

        cumRMSE = 0
        times = self.times
        #Run many times - just to get rid of randomness in assigning random weights
        for i in range(times):

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
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE / times

        #Return the error
        #print("\nOptimizing connectivity..")
        #print("Regression error"+str(regressionError)+"\n")
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
        #print("\nOptimizing esn parameters..")
        #print("Regression error"+str(regressionError)+"\n")
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
        #print("Optimizing connectivity..")
        #print("\nRegression error:"+str(regressionError)+"\n")
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
        #print("Optimizing spectral radius and leaking rate...")
        #print("\nRegression error"+str(regressionError)+"\n")
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


# class ESNConnTuner:
#     def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
#                  inputConnectivityBound, reservoirConnectivityBound, times= 10):
#         self.size = size
#         self.initialTransient = initialTransient
#         self.trainingInputData = trainingInputData
#         self.trainingOutputData = trainingOutputData
#         self.validationInputData = validationInputData
#         self.validationOutputData = validationOutputData
#         self.inputConnectivityBound = inputConnectivityBound
#         self.reservoirConnectivityBound = reservoirConnectivityBound
#         self.times = times
#
#
#     def __ESNTrain__(self, x):
#
#         #Extract the parameters
#         inputConnectivity = x[0]
#         reservoirConnectivity = x[1]
#         reservoirTopology = topology.RandomTopology(size=self.size, connectivity=reservoirConnectivity)
#
#         cumRMSE = 0
#         times = self.times
#         for i in range(times):
#             #Create the network
#             esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
#                                                     inputData=self.trainingInputData,
#                                                     outputData=self.trainingOutputData,
#                                                     reservoirTopology=reservoirTopology,
#                                                     inputConnectivity=inputConnectivity
#                                                     )
#
#             #Train the reservoir
#             esn.trainReservoir()
#
#             #Predict for the validation data
#             predictedOutputData = esn.predict(self.validationInputData)
#
#             #Calcuate the regression error
#             errorFunction = rmse.RootMeanSquareError()
#             regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
#             cumRMSE += regressionError
#
#             #Free the memory
#             gc.collect()
#
#         regressionError = cumRMSE / times
#         #Return the error
#         #print("Optimizing connectivity..")
#         print("\nRegression error:"+str(regressionError)+"\n")
#         return regressionError
#
#     def __tune__(self):
#         bounds = [self.inputConnectivityBound, self.reservoirConnectivityBound]
#         result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
#         print("\nThe optimal parameters for classic ESN:"+str(result.x))
#         return result.x[0], result.x[1]
#
#     def getOptimalParameters(self):
#         return self.__tune__()


# class ESNErdosTuner:
#     def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
#                  inputConnectivityBound, probabilityBound, times=10):
#         self.size = size
#         self.initialTransient = initialTransient
#         self.trainingInputData = trainingInputData
#         self.trainingOutputData = trainingOutputData
#         self.validationInputData = validationInputData
#         self.validationOutputData = validationOutputData
#         self.inputConnectivityBound = inputConnectivityBound
#         self.probabilityBound = probabilityBound
#         self.times = times
#
#
#     def __ESNTrain__(self, x):
#
#         #Extract the parameters
#         inputConnectivity = x[0]
#         probability = x[1]
#         reservoirTopology = topology.ErdosRenyiTopology(size=self.size, probability=probability)
#         print("\nInput:"+str(inputConnectivity)+" Probability:"+str(probability))
#
#         cumRMSE = 0
#         times = self.times
#         for i in range(times):
#             #Create the network
#             esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
#                                                     inputData=self.trainingInputData,
#                                                     outputData=self.trainingOutputData,
#                                                     reservoirTopology=reservoirTopology,
#                                                     inputConnectivity=inputConnectivity
#                                                     )
#
#             #Train the reservoir
#             esn.trainReservoir()
#
#             #Predict for the validation data
#             predictedOutputData = esn.predict(self.validationInputData)
#
#             #Calcuate the regression error
#             errorFunction = rmse.RootMeanSquareError()
#             regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
#             cumRMSE += regressionError
#
#             #Free the memory
#             gc.collect()
#
#         regressionError = cumRMSE/times
#         #Return the error
#         #print("Optimizing connectivity..")
#         print("\nRegression error:"+str(regressionError)+"\n")
#         return regressionError
#
#     def __tune__(self):
#         bounds = [self.inputConnectivityBound, self.probabilityBound]
#         result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
#         print("\nThe optimal parameters for Erdos ESN:"+str(result.x))
#         return result.x[0], result.x[1]
#
#     def getOptimalParameters(self):
#         return self.__tune__()



class ESNErdosFullTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, inputScalingBound, reservoirScalingBound, leakingRateBound, inputConnectivityBound, probabilityBound, times=10):
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
        self.probabilityBound = probabilityBound
        self.times = times

        self.inputWeightConn = None
        self.reservoirWeightConn = None

    def __ESNConnTrain__(self, x):
        #Extract the parameters
        inputConnectivity = x[0]
        probability = x[1]
        reservoirTopology = topology.ErdosRenyiTopology(size=self.size, probability=probability)
        #print("\nInput:"+str(inputConnectivity)+" Probability:"+str(probability))

        cumRMSE = 0
        times = 10
        #Run many times - just to get rid of randomness in assigning random weights
        for i in range(times):

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
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE / times

        #Return the error
        #print("\nOptimizing connectivity..")
        #print("Regression error"+str(regressionError)+"\n")
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
                                                reservoirTopology=topology.ErdosRenyiTopology(self.size, self.probabilityOptimum))

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
        #t("\nOptimizing esn parameters..")
        #print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):

        # First tune for the input connectivity and the reservoir connectivity
        connBounds = [self.inputConnectivityBound,self.probabilityBound]
        connResult = optimize.differential_evolution(self.__ESNConnTrain__,bounds=connBounds)
        self.inputConnectivityOptimum = connResult.x[0]
        self.probabilityOptimum = connResult.x[1]

        # With tuned parameters, create the network with optimal connections and keep the connections as same
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=topology.ErdosRenyiTopology(size=self.size,
                                                                                          probability=self.probabilityOptimum),
                                                inputConnectivity=self.inputConnectivityOptimum)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

        # Tune the other parameters
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

# class ESNSmallWorldGraphsTuner:
#     def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
#                  inputConnectivityBound, meanDegreeBound, betaBound, times=10):
#         self.size = size
#         self.initialTransient = initialTransient
#         self.trainingInputData = trainingInputData
#         self.trainingOutputData = trainingOutputData
#         self.validationInputData = validationInputData
#         self.validationOutputData = validationOutputData
#         self.inputConnectivityBound = inputConnectivityBound
#         self.meanDegreeBound = meanDegreeBound
#         self.betaBound = betaBound
#         self.times = times
#
#
#     def __ESNTrain__(self, x):
#
#         #Extract the parameters
#         inputConnectivity = x[0]
#         meanDegree = int(np.floor(x[1]))
#         beta = x[2]
#         reservoirTopology = topology.SmallWorldGraphs(size=self.size, meanDegree=meanDegree, beta=beta)
#         #print("\nOptimizing connectivity..")
#         print("\nParameters:"+str(x))
#
#         cumRMSE = 0
#         times = self.times
#         for i in range(times):
#             #Create the network
#             esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
#                                                     inputData=self.trainingInputData,
#                                                     outputData=self.trainingOutputData,
#                                                     reservoirTopology=reservoirTopology,
#                                                     inputConnectivity=inputConnectivity
#                                                     )
#
#             #Train the reservoir
#             esn.trainReservoir()
#
#             #Predict for the validation data
#             predictedOutputData = esn.predict(self.validationInputData)
#
#             #Calcuate the regression error
#             errorFunction = rmse.RootMeanSquareError()
#             regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
#             cumRMSE += regressionError
#
#             #Free the memory
#             gc.collect()
#
#         regressionError = cumRMSE / times
#         #Return the error
#         print("\nRegression error:"+str(regressionError)+"\n")
#         return regressionError
#
#     def __tune__(self):
#         bounds = [self.inputConnectivityBound, self.meanDegreeBound, self.betaBound]
#         result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
#         print("\nThe optimal parameters for Small World Graphs ESN:"+str(result.x))
#         return result.x[0], int(np.floor(result.x[1])), result.x[2]
#
#     def getOptimalParameters(self):
#         return self.__tune__()

class ESNSmallWorldGraphsFullTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 spectralRadiusBound, inputScalingBound, reservoirScalingBound, leakingRateBound, inputConnectivityBound, meanDegreeBound, betaBound, times=10):
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
        self.meanDegreeBound = meanDegreeBound
        self.betaBound = betaBound
        self.times = times

        self.inputWeightConn = None
        self.reservoirWeightConn = None

    def __ESNConnTrain__(self, x):
        #Extract the parameters
        inputConnectivity = x[0]
        meanDegree = int(np.floor(x[1]))
        beta = x[2]
        reservoirTopology = topology.SmallWorldGraphs(size=self.size, meanDegree=meanDegree, beta=beta)
        #print("\nParameters:"+str(x))

        cumRMSE = 0
        times = 10
        #Run many times - just to get rid of randomness in assigning random weights
        for i in range(times):

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
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE / times

        #Return the error
        #print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __ESNTrain__(self, x):
        #print("\nOptimizing esn parameters:"+str(x))
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
                                                reservoirTopology=topology.SmallWorldGraphs(size=self.size,
                                                                                            meanDegree=self.meanDegreeOptimum,
                                                                                            beta=self.betaOptimum))

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
        #print("Regression error"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):

        # First tune for the input connectivity and the reservoir connectivity
        connBounds = [self.inputConnectivityBound,self.meanDegreeBound, self.betaBound]
        connResult = optimize.differential_evolution(self.__ESNConnTrain__,bounds=connBounds)
        self.inputConnectivityOptimum = connResult.x[0]
        self.meanDegreeOptimum = int(np.floor(connResult.x[1]))
        self.betaOptimum = connResult.x[2]

        # With tuned parameters, create the network with optimal connections and keep the connections as same
        esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                inputData=self.trainingInputData,
                                                outputData=self.trainingOutputData,
                                                reservoirTopology=topology.SmallWorldGraphs(size=self.size,
                                                                                            meanDegree=self.meanDegreeOptimum,
                                                                                            beta=self.betaOptimum),
                                                inputConnectivity=self.inputConnectivityOptimum)
        self.inputWeightConn = esn.inputWeightRandom, esn.randomInputIndices
        self.reservoirWeightConn = esn.reservoirWeightRandom, esn.randomReservoirIndices

        # Tune the other parameters
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()

# class ESNScaleFreeNetworksTuner:
#     def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
#                  inputConnectivityBound, attachmentBound, times=10):
#         self.size = size
#         self.initialTransient = initialTransient
#         self.trainingInputData = trainingInputData
#         self.trainingOutputData = trainingOutputData
#         self.validationInputData = validationInputData
#         self.validationOutputData = validationOutputData
#         self.inputConnectivityBound = inputConnectivityBound
#         self.attachmentBound = attachmentBound
#         self.times = times
#
#
#     def __ESNTrain__(self, x):
#
#         #Extract the parameters
#         inputConnectivity = x[0]
#         attachment = int(np.floor(x[1]))
#
#         reservoirTopology = topology.ScaleFreeNetworks(size=self.size, attachmentCount=attachment)
#         #print("\nOptimizing connectivity..")
#         print("\nInput Connectivity:"+str(inputConnectivity)+ " Attachment:"+str(attachment))
#
#         cumRMSE = 0
#         times = self.times
#         for i in range(times):
#             #Create the network
#             esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
#                                                     inputData=self.trainingInputData,
#                                                     outputData=self.trainingOutputData,
#                                                     reservoirTopology=reservoirTopology,
#                                                     inputConnectivity=inputConnectivity
#                                                     )
#
#             #Train the reservoir
#             esn.trainReservoir()
#
#             #Predict for the validation data
#             predictedOutputData = esn.predict(self.validationInputData)
#
#             #Calcuate the regression error
#             errorFunction = rmse.RootMeanSquareError()
#             regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
#             cumRMSE += regressionError
#
#             #Free the memory
#             gc.collect()
#
#         regressionError = cumRMSE / times
#         #Return the error
#         print("\nRegression error:"+str(regressionError)+"\n")
#         return regressionError
#
#     def __tune__(self):
#         bounds = [self.inputConnectivityBound, self.attachmentBound]
#         result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
#         print("\nThe optimal parameters for Scale Free Networks ESN:"+str(result.x))
#         return result.x[0], int(np.floor(result.x[1]))
#
#     def getOptimalParameters(self):
#         return self.__tune__()


class ESNScaleFreeNetworksTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 attachmentBound, times=10, inputConnectivity=0.6):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.inputConnectivity = inputConnectivity
        self.attachmentBound = attachmentBound
        self.times = times


    def __ESNTrain__(self, x):

        #Extract the parameters
        attachment = int(np.floor(x[0]))

        reservoirTopology = topology.ScaleFreeNetworks(size=self.size, attachmentCount=attachment)
        #print("\nOptimizing connectivity..")
        print("\nAttachment:"+str(attachment))

        cumRMSE = 0
        times = self.times
        for i in range(times):
            #Create the network
            esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                    inputData=self.trainingInputData,
                                                    outputData=self.trainingOutputData,
                                                    reservoirTopology=reservoirTopology,
                                                    inputConnectivity=self.inputConnectivity
                                                    )

            #Train the reservoir
            esn.trainReservoir()

            #Predict for the validation data
            predictedOutputData = esn.predict(self.validationInputData)

            #Calcuate the regression error
            errorFunction = rmse.RootMeanSquareError()
            regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE / times
        #Return the error
        print("\nRegression error:"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.attachmentBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        #print("\nThe optimal parameters for Scale Free Networks ESN:"+str(result.x))
        return int(np.floor(result.x[0]))

    def getOptimalParameters(self):
        return self.__tune__()

class ESNSmallWorldGraphsTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 meanDegreeBound, betaBound, times=10, inputConnectivity=0.6):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.inputConnectivity = inputConnectivity
        self.meanDegreeBound = meanDegreeBound
        self.betaBound = betaBound
        self.times = times


    def __ESNTrain__(self, x):

        #Extract the parameters
        meanDegree = int(np.floor(x[0]))
        beta = x[1]
        reservoirTopology = topology.SmallWorldGraphs(size=self.size, meanDegree=meanDegree, beta=beta)
        #print("\nOptimizing connectivity..")
        print("\nParameters:"+str(x))

        cumRMSE = 0
        times = self.times
        for i in range(times):
            #Create the network
            esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                    inputData=self.trainingInputData,
                                                    outputData=self.trainingOutputData,
                                                    reservoirTopology=reservoirTopology,
                                                    inputConnectivity=self.inputConnectivity
                                                    )

            #Train the reservoir
            esn.trainReservoir()

            #Predict for the validation data
            predictedOutputData = esn.predict(self.validationInputData)

            #Calcuate the regression error
            errorFunction = rmse.RootMeanSquareError()
            regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE / times
        #Return the error
        #print("\nRegression error:"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.meanDegreeBound, self.betaBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        print("\nThe optimal parameters for Small World Graphs ESN:"+str(result.x))
        return int(np.floor(result.x[0])), result.x[1]

    def getOptimalParameters(self):
        return self.__tune__()

class ESNErdosTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 probabilityBound, times=10, inputConnectivity=0.6):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.inputConnectivity = inputConnectivity
        self.probabilityBound = probabilityBound
        self.times = times


    def __ESNTrain__(self, x):

        #Extract the parameters
        probability = x[0]
        reservoirTopology = topology.ErdosRenyiTopology(size=self.size, probability=probability)
        print("\nProbability:"+str(probability))

        cumRMSE = 0
        times = self.times
        for i in range(times):
            #Create the network
            esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                    inputData=self.trainingInputData,
                                                    outputData=self.trainingOutputData,
                                                    reservoirTopology=reservoirTopology,
                                                    inputConnectivity=self.inputConnectivity
                                                    )

            #Train the reservoir
            esn.trainReservoir()

            #Predict for the validation data
            predictedOutputData = esn.predict(self.validationInputData)

            #Calcuate the regression error
            errorFunction = rmse.RootMeanSquareError()
            regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE/times
        #Return the error
        #print("Optimizing connectivity..")
        #print("\nRegression error:"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.probabilityBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        print("\nThe optimal parameters for Erdos ESN:"+str(result.x))
        return result.x[0]

    def getOptimalParameters(self):
        return self.__tune__()

class ESNConnTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                 reservoirConnectivityBound, times=10,  inputConnectivity=0.6):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.inputConnectivity = inputConnectivity
        self.reservoirConnectivityBound = reservoirConnectivityBound
        self.times = times


    def __ESNTrain__(self, x):

        #Extract the parameters
        reservoirConnectivity = x[0]
        reservoirTopology = topology.RandomTopology(size=self.size, connectivity=reservoirConnectivity)
        print("\nThe parameter:"+str(x))
        cumRMSE = 0
        times = self.times
        for i in range(times):
            #Create the network
            esn = EchoStateNetwork.EchoStateNetwork(size=self.size,
                                                    inputData=self.trainingInputData,
                                                    outputData=self.trainingOutputData,
                                                    reservoirTopology=reservoirTopology,
                                                    inputConnectivity=self.inputConnectivity
                                                    )

            #Train the reservoir
            esn.trainReservoir()

            #Predict for the validation data
            predictedOutputData = esn.predict(self.validationInputData)

            #Calcuate the regression error
            errorFunction = rmse.RootMeanSquareError()
            regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)
            cumRMSE += regressionError

            #Free the memory
            gc.collect()

        regressionError = cumRMSE / times
        #Return the error
        #print("Optimizing connectivity..")
        print("\nRegression error:"+str(regressionError)+"\n")
        return regressionError

    def __tune__(self):
        bounds = [self.reservoirConnectivityBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        print("\nThe optimal parameters for classic ESN:"+str(result.x))
        return result.x[0]

    def getOptimalParameters(self):
        return self.__tune__()