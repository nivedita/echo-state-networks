from scipy import optimize
from reservoir import EchoStateNetwork
from performance import RootMeanSquareError as rmse
import numpy as np

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

        #Return the error
        return regressionError

    def __tune__(self):
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__ESNTrain__,bounds=bounds)
        return result.x[0], result.x[1], result.x[2], result.x[3], self.inputWeightConn, self.reservoirWeightConn

    def getOptimalParameters(self):
        return self.__tune__()