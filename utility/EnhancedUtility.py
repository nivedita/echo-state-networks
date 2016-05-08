import numpy as np
from enum import Enum
from reservoir import classicESN as esn
import pandas as pd
import gc
from performance import ErrorMetrics as metrics
from scipy import optimize

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
                 initialSeedSeries, validationOutputData, arbitraryDepth,
                 spectralRadiusBound, inputScalingBound,
                 reservoirScalingBound, leakingRateBound,inputWeightMatrix=None,
                 reservoirWeightMatrix=None, minimizer=Minimizer.DifferentialEvolution,
                 initialGuess = np.array([0.79, 0.5, 0.5, 0.3])):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeedSeries = initialSeedSeries
        self.validationOutputData = validationOutputData
        self.arbitraryDepth = arbitraryDepth
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

        # Create the reservoir
        res = esn.Reservoir(size=self.size,
                                  spectralRadius=spectralRadius,
                                  inputScaling=inputScaling,
                                  reservoirScaling=reservoirScaling,
                                  leakingRate=leakingRate,
                                  initialTransient=self.initialTransient,
                                  inputData=self.trainingInputData,
                                  outputData=self.trainingOutputData,
                                  inputWeightRandom=self.inputWeightRandom,
                                  reservoirWeightRandom=self.reservoirWeightRandom)

        # Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        # Predict for the validation data
        predictedOutputData = self.predict(res, self.initialSeedSeries, self.arbitraryDepth, self.horizon)

        gc.collect()

        # Calcuate the regression error
        errorFunction = metrics.MeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def predict(self, network, availableSeries, arbitraryDepth, horizon):
        # To avoid mutation of pandas series
        initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
        for i in range(horizon):
            feature = initialSeries.values[-arbitraryDepth:].reshape((1, arbitraryDepth))

            # Append bias
            feature = np.hstack((1.0,feature[0, :])).reshape((1, feature.shape[1]+1))

            nextPoint = network.predictOnePoint(feature)[0]

            nextIndex = initialSeries.last_valid_index() + pd.Timedelta(days=1)
            initialSeries[nextIndex] = nextPoint

        predictedSeries = initialSeries[-horizon:]
        return predictedSeries


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
