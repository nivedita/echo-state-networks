from scipy import optimize
from reservoir import Reservoir
from performance import RootMeanSquareError as rmse
import numpy as np

class ReservoirTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData, spectralRadiusBound, inputScalingBound, leakingRateBound):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.spectralRadiusBound = spectralRadiusBound
        self.inputScalingBound = inputScalingBound
        self.leakingRateBound = leakingRateBound

         #Generate the random input and reservoir matrices
        self.inputN, self.inputD = self.trainingInputData.shape
        self.inputWeightRandom = np.random.rand(self.size, self.inputD)
        self.reservoirWeightRandom = np.random.rand(self.size, self.size)


    def __reservoirTrain__(self, x):

        #Extract the parameters
        spectralRadius = x[0]
        inputScaling = x[1]
        leakingRate = x[2]
        #Create the reservoir
        res = Reservoir.Reservoir(size=self.size, spectralRadius=spectralRadius, inputScaling=inputScaling, leakingRate=leakingRate, initialTransient=self.initialTransient,inputData=self.trainingInputData, outputData=self.trainingOutputData, inputWeightRandom=self.inputWeightRandom, reservoirWeightRandom=self.reservoirWeightRandom)

        #Train the reservoir
        res.trainReservoir()

        #Predict for the validation data
        predictedOutputData = res.predict(self.validationInputData)

        #Calcuate the regression error
        errorFunction = rmse.RootMeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        return regressionError

    def __tune__(self):
        bounds = [self.spectralRadiusBound, self.inputScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__reservoirTrain__,bounds=bounds)
        return result.x[0], result.x[1], result.x[2], self.inputWeightRandom, self.reservoirWeightRandom

    def getOptimalParameters(self):
        return self.__tune__()