import numpy as np
from GAOptimizer import Optimizer as opt, MutationOperators as mOperators, CrossoverOperators as cOperators
from reservoir import classicESN as esn, Utility as util
from performance import ErrorMetrics as metrics
import gc


class ESNParameterErrorObjective():

    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationOutputData,
                 inputWeight, reservoirWeight, initialSeed, horizon):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationOutputData = validationOutputData
        self.inputWeightRandom = inputWeight
        self.reservoirWeightRandom = reservoirWeight
        self.initialSeed = initialSeed
        self.horizon = horizon


    def evaluate(self, x):

        #Extract the parameters
        spectralRadius = x[0, 0]
        inputScaling = x[0, 1]
        reservoirScaling = x[0, 2]
        leakingRate = x[0, 3]

        #Create the reservoir
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

        #Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        #Predict for the validation data
        predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

        gc.collect()

        #Calcuate the regression error
        errorFunction = metrics.MeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        return regressionError


class ReservoirParameterTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeed, validationOutputData, spectralRadiusBound, inputScalingBound,
                 reservoirScalingBound, leakingRateBound,inputWeightMatrix=None,
                 reservoirWeightMatrix=None):
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


        if inputWeightMatrix is None:
            self.inputN, self.inputD = self.trainingInputData.shape
            self.inputWeightRandom = np.random.rand(self.size, self.inputD)
        else:
            self.inputWeightRandom = inputWeightMatrix

        if reservoirWeightMatrix is None:
            self.reservoirWeightRandom = np.random.rand(self.size, self.size)
        else:
            self.reservoirWeightRandom = reservoirWeightMatrix

        # Make the input and reservoir weight matrices read only - This is needed just to be safe (We do not need mutation of numpy arrays)
        self.inputWeightRandom.flags.writeable = False
        self.reservoirWeightRandom.flags.writeable = False


    def evaluate(self, x):

        #Extract the parameters
        spectralRadius = x[0, 0]
        inputScaling = x[0, 1]
        reservoirScaling = x[0, 2]
        leakingRate = x[0, 3]

        #Create the reservoir
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

        #Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        #Predict for the validation data
        predictedOutputData = util.predictFuture(res, self.initialSeed, self.horizon)

        gc.collect()

        #Calcuate the regression error
        errorFunction = metrics.MeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        return regressionError

    def __tune__(self):

        # Create the GA Optimizer
        objective = ESNParameterErrorObjective(size=self.size, initialTransient=self.initialTransient,
                                               trainingInputData=self.trainingInputData,
                                               trainingOutputData=self.trainingOutputData,
                                               validationOutputData=self.validationOutputData,
                                               inputWeight=self.inputWeightRandom,
                                               reservoirWeight=self.reservoirWeightRandom,
                                               initialSeed=self.initialSeed,
                                               horizon=self.horizon)

        optimizer = opt.Optimizer(populationSize=100,
                                  parameterBounds=[self.spectralRadiusBound, self.inputScalingBound,
                                                   self.reservoirScalingBound, self.leakingRateBound],
                                  fitnessObj=objective,
                                  crossoverOperator=cOperators.TwoPointAndLineHybridCrossover(),
                                  mutationOperator=mOperators.SimpleAndUniformHybrid(),
                                  maxGeneration=100,
                                  mutationRate=0.1,
                                  elitismRate=0.3)

        optimizer.optimize()

        # Return the optimal parameters
        result = optimizer.getOptimalParameters()[0][0].tolist()
        spectralRadiusOptimal, inputScalingOptimal, reservoirScalingOptimum, leakingRateOptimal = tuple(result)
        return spectralRadiusOptimal, inputScalingOptimal, reservoirScalingOptimum, leakingRateOptimal

    def getOptimalParameters(self):
        return self.__tune__()