from scipy import optimize
from reservoir import EchoStateNetwork, DetermimisticReservoir as dr
from performance import RootMeanSquareError as rmse

class DeterministicReservoirTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData, validationInputData, validationOutputData, inputWeight_v, reservoirTopology, inputScalingBound, leakingRateBound):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.validationInputData = validationInputData
        self.validationOutputData = validationOutputData
        self.inputWeight_v = inputWeight_v
        self.inputScalingBound = inputScalingBound
        self.leakingRateBound = leakingRateBound
        self.reservoirTopology = reservoirTopology



    def __reservoirTrain__(self, x):

        #Extract the parameters
        inputScaling = x[0]
        leakingRate = x[1]

        #Create the reservoir
        res = dr.DeterministicReservoir(size=self.size,
                                        inputWeight_v=self.inputWeight_v,
                                        inputWeightScaling=inputScaling,
                                        inputData=self.trainingInputData,
                                        outputData=self.trainingOutputData,
                                        leakingRate=leakingRate,
                                        initialTransient=self.initialTransient,
                                        reservoirTopology=self.reservoirTopology)
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
        bounds = [self.inputScalingBound, self.leakingRateBound]
        result = optimize.differential_evolution(self.__reservoirTrain__,bounds=bounds)
        return result.x[0], result.x[1]

    def getOptimalParameters(self):
        return self.__tune__()