from scipy import optimize
from reservoir import ClassicESN
from performance import RootMeanSquareError as rmse
import numpy as np
from reservoir import Utility as util
from enum import Enum

class Minimizer(Enum):
    BasinHopping = 1
    DifferentialEvolution = 2

class ParameterStep(object):
    def __init__(self, stepsize=0.01):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        x += 0.01
        return x

class ReservoirTuner:
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
        res = ClassicESN.Reservoir(size=self.size,
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

        #Predict for the validation data
        predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

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
            return result.x[0], result.x[1], result.x[2], result.x[3]
        else:
            bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
            minimizer_kwargs = {"method": "TNC", "bounds":bounds, "options": {"eps":0.05}}
            mytakestep = ParameterStep()
            result = optimize.basinhopping(self.__reservoirTrain__, x0=self.initialGuess, minimizer_kwargs=minimizer_kwargs, take_step=mytakestep, stepsize=0.01)
            return result.x[0], result.x[1], result.x[2], result.x[3]

    def getOptimalParameters(self):
        return self.__tune__()