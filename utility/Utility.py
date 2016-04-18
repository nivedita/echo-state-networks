import pandas as pd
import numpy as np
from timeseries import TimeSeriesContinuousProcessor as processor, TimeSeriesInterval as tsi
from reservoir import classicESN as esn, onlineESNWithRLS as onlineESN, ReservoirTopology as topology, ActivationFunctions as act
from sklearn import preprocessing as pp
import os
from plotting import OutputTimeSeries as plotting, LineBarCombinationPlot as combPlotting
# from reservoir import Tuner as tuner
from performance import ErrorMetrics as metrics
from enum import Enum
import gc
from scipy import optimize
from scipy.stats import pearsonr, spearmanr
import operator

class Minimizer(Enum):
    BasinHopping = 1
    DifferentialEvolution = 2
    BruteForce = 3

class FeatureSelectionMethod:
    CutOff_Threshold = 1
    Pattern_Analysis = 2

class LearningMethod:
    Batch = 1
    Online = 2

class ParameterStep(object):
    def __init__(self, stepsize=0.005):
        self.stepsize = stepsize
    def __call__(self, x):
        x += self.stepsize
        return x

class ReservoirParameterTuner:
    def __init__(self, size, initialTransient, trainingInputData, trainingOutputData,
                 initialSeedSeries, validationOutputData, arbitraryDepth, featureIndices,
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
        self.featureIndices = featureIndices
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
        res = esn.Reservoir(size=self.size,
                                  spectralRadius=spectralRadius,
                                  inputScaling=inputScaling,
                                  reservoirScaling=reservoirScaling,
                                  leakingRate=leakingRate,
                                  initialTransient=self.initialTransient,
                                  inputData=self.trainingInputData,
                                  outputData=self.trainingOutputData,
                                  inputWeightRandom=self.inputWeightRandom,
                                  reservoirWeightRandom=self.reservoirWeightRandom,
                                  activationFunction=esn.ActivationFunction.EXPIT,
                                  outputRelu=True)

        #Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        # Predict for the validation data
        predictedOutputData = self.predict(res, self.initialSeedSeries, self.arbitraryDepth, self.horizon, self.featureIndices)

        gc.collect()

        #Calcuate the regression error
        errorFunction = metrics.MeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def predict(self, network, availableSeries, arbitraryDepth, horizon, featureIndices):
        # To avoid mutation of pandas series
        initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
        for i in range(horizon):
            feature = initialSeries.values[-arbitraryDepth:].reshape((1, arbitraryDepth))
            feature = feature[:, featureIndices]

            #Append bias
            feature = np.hstack((1.0,feature[0, :])).reshape((1, feature.shape[1]+1))

            nextPoint = network.predictOnePoint(feature)[0]

            nextIndex = initialSeries.last_valid_index() + pd.Timedelta(hours=1)
            initialSeries[nextIndex] = nextPoint

        predictedSeries = initialSeries[-horizon:]
        return predictedSeries


    def __tune__(self):
        if self.minimizer == Minimizer.DifferentialEvolution:
            bounds = [self.spectralRadiusBound, self.inputScalingBound, self.reservoirScalingBound, self.leakingRateBound]
            result = optimize.differential_evolution(self.__reservoirTrain__,bounds=bounds, maxiter=1)
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

class CorrelationBruteTuner:
    def __init__(self, size, initialTransient, trainingSeries, featureVectors, targetVectors, correlationCoefficients,
                 validationSeries, arbitraryDepth,
                 reservoirActivationFunction, outputActivationFunction,
                 inputConnMatrix, reservoirWeightMatrix,
                 spectralRadius=0.79, inputScaling=0.0, reservoirScaling=0.0
                 ):
        self.size = size
        self.initialTransient = initialTransient
        self.util = SeriesUtility()
        self.trainingSeries = trainingSeries
        self.trainingInputData = featureVectors
        self.trainingOutputData = targetVectors
        self.validationSeries = validationSeries
        self.horizon = self.validationSeries.values.shape[0]
        self.correlationCoefficients = correlationCoefficients
        self.depth = arbitraryDepth
        self.reservoirActivationFunction = reservoirActivationFunction
        self.outputActivationFunction = outputActivationFunction
        self.inputConnMatrix = inputConnMatrix
        self.reservoirWeightMatrix = reservoirWeightMatrix

        # Input-to-reservoir is of Classic Type - Fully connected and maintained as constant
        self.inputN, self.inputD = self.trainingInputData.shape

        # Other reservoir parameters are also kept constant
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling

        # Scaling and leaking rate range
        self.ranges = (slice(0.0, 1.10, 0.25), slice(0.3,1.0,0.2))


    def predict(self, network, availableSeries, arbitraryDepth, horizon):
        # To avoid mutation of pandas series
        initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
        for i in range(horizon):
            feature = initialSeries.values[-arbitraryDepth:].reshape((1, arbitraryDepth))

            # Append bias
            feature = np.hstack((1.0,feature[0, :])).reshape((1, feature.shape[1]+1))

            nextPoint = network.predictOnePoint(feature)[0]

            nextIndex = initialSeries.last_valid_index() + pd.Timedelta(hours=1)
            initialSeries[nextIndex] = nextPoint

        predictedSeries = initialSeries[-horizon:]
        return predictedSeries

    def __reservoirTrain__(self, x):

        # Extract the parameters
        correlationScaling = float(x[0])
        leakingRate = float(x[1])

        correlationMatrix = self.util.getScaleCorrelationMatrix(self.correlationCoefficients, correlationScaling,
                                                                self.inputD, self.size)

        inputWeightMatrix = self.inputConnMatrix * correlationMatrix

        #Create the reservoir
        res = esn.Reservoir(size=self.size,
                            spectralRadius=self.spectralRadius,
                            inputScaling=self.inputScaling,
                            reservoirScaling=self.reservoirScaling,
                            leakingRate=leakingRate,
                            initialTransient=self.initialTransient,
                            inputData=self.trainingInputData,
                            outputData=self.trainingOutputData,
                            inputWeightRandom=inputWeightMatrix,
                            reservoirWeightRandom=self.reservoirWeightMatrix,
                            reservoirActivationFunction=self.reservoirActivationFunction,
                            outputActivationFunction=self.outputActivationFunction)

        # Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        # Predict for the validation data
        predictedSeries = self.predict(res, self.trainingSeries, self.depth, self.horizon)

        gc.collect()

        # Calcuate the regression error
        errorFunction = metrics.MeanSquareError()
        error = errorFunction.compute(self.validationSeries.values, predictedSeries.values)

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(error))
        return error

    def __tune__(self):
        result = optimize.brute(self.__reservoirTrain__,ranges=self.ranges, finish=None, full_output=True)
        return result[0]

    def getOptimalParameters(self):
        return self.__tune__()

class SeriesUtility:
    def __init__(self):
        self.esn = None
        self.scalingFunction = None

    def convertDatasetsToSeries(self, fileName):
        # Read the data
        df = pd.read_csv(fileName, index_col=0, parse_dates=True, names=['value'])

        # Convert the dataframe into series
        data = np.array(df.as_matrix()).flatten().astype(float)
        series = pd.Series(data=data, index=df.index)
        return series

    def intersect(self, seriesList):
        index = seriesList[0].index

        for i in range(1,len(seriesList)):
            index = seriesList[i].index.intersection(index)

        intersection = []
        for series in seriesList:
            intersection.append(pd.Series(data=series[index], index=index))
        return intersection

    def _sum(self, x):
        if len(x) == 0:
            return 0
        else:
            return sum(x)

    def _exists(self, x):
        if len(x) == 0:
            return 0
        else:
            return 1

    def _binary(self, x):
        if len(x) == 0:
            return -1
        else:
            return +1

    def _mean(self, x):
        if len(x) == 0:
            return 0
        else:
            return sum(x)/len(x)

    def resampleSeriesExists(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._exists)

    def resampleSeriesSum(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._sum)

    def resampleSeriesBinary(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._binary)

    def resampleSeriesMean(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._mean)

    def scaleSeries(self, series):
        self.scalingFunction = pp.MinMaxScaler((0,1))
        #self.scalingFunction = pp.MinMaxScaler((-1,1))
        #self.scalingFunction = pp.StandardScaler()
        data = series.values
        data = self.scalingFunction.fit_transform(data)
        scaledSeries = pd.Series(data=data,index=series.index)
        return scaledSeries

    def scaleSeriesStandard(self, series):
        #self.scalingFunction = pp.MinMaxScaler((0,1))
        #self.scalingFunction = pp.MinMaxScaler((-1,1))
        self.scalingFunction = pp.StandardScaler()
        data = series.values
        data = self.scalingFunction.fit_transform(data)
        scaledSeries = pd.Series(data=data,index=series.index)
        return scaledSeries

    def descaleSeries(self, series):
        data = series.values
        data = self.scalingFunction.inverse_transform(data)
        descaledSeries = pd.Series(data=data, index=series.index)
        return descaledSeries

    def removeNegativeValues(self, series):
        data = series.values
        data[data < 0.0] = 0.0
        positiveSeries = pd.Series(data=data, index=series.index)
        return positiveSeries

    def splitIntoTrainingAndTestingSeries(self, series, horizon):
        index = series.index
        data = series.values
        trainingData = data[:data.shape[0]-horizon]
        trainingIndex = index[:index.shape[0]-horizon]
        testingData = data[data.shape[0]-horizon:]
        testingIndex = index[index.shape[0]-horizon:]

        trainingSeries = pd.Series(data=trainingData.flatten(),index=trainingIndex)
        testingSeries = pd.Series(data=testingData.flatten(),index=testingIndex)
        return trainingSeries, testingSeries

    def splitIntoTrainingAndValidationSeries(self, series, trainingSetRatio):
        index = series.index
        data = series.values
        splitIndex = int(trainingSetRatio * data.shape[0])
        trainingData = data[:splitIndex]
        trainingIndex = index[:splitIndex]
        validationData = data[splitIndex:]
        validationIndex = index[splitIndex:]

        trainingSeries = pd.Series(data=trainingData.flatten(),index=trainingIndex)
        validationSeries = pd.Series(data=validationData.flatten(),index=validationIndex)
        return trainingSeries, validationSeries

    def formFeatureAndTargetVectors(self, series, depth):
        # Feature list
        featureIntervalList = []
        for i in range(depth, 0, -1):
            interval = pd.Timedelta(hours=-(i))
            featureIntervalList.append(interval)

        # Target vectors
        targetIntervalList = [pd.Timedelta(hours=0)]

        # Pre-process the data and form feature and target vectors
        tsp = tsi.TimeSeriesIntervalProcessor(series, featureIntervalList, targetIntervalList)
        featureVectors, targetVectors = tsp.getProcessedData()

        # Append bias to feature vectors
        featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)), featureVectors))

        return featureVectors, targetVectors

    def formFeatureAndTargetVectorsInterval(self, series, depth, period):
        # Feature list
        featureIntervalList = []
        for i in range(period, 0, -1):
            interval = pd.Timedelta(hours=-(i * depth))
            featureIntervalList.append(interval)

        # Target vectors
        targetIntervalList = [pd.Timedelta(hours=0)]

        # Pre-process the data and form feature and target vectors
        tsp = tsi.TimeSeriesIntervalProcessor(series, featureIntervalList, targetIntervalList)
        featureVectors, targetVectors = tsp.getProcessedData()

        # Append bias to feature vectors
        featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)), featureVectors))

        return featureVectors, targetVectors


    def formContinousFeatureAndTargetVectors(self, series, depth):
        # Pre-process the data and form feature and target vectors
        tsp = processor.TimeSeriesContinuosProcessor(series, depth, horizon=1)
        featureVectors, targetVectors = tsp.getProcessedData()

        # Append bias to feature vectors
        featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)), featureVectors))

        return featureVectors, targetVectors

    def formContinousFeatureAndTargetVectorsWithoutBias(self, series, depth):
        # Pre-process the data and form feature and target vectors
        tsp = processor.TimeSeriesContinuosProcessor(series, depth, horizon=1)
        featureVectors, targetVectors = tsp.getProcessedData()

        return featureVectors, targetVectors


    def trainESNWithoutTuning(self, size, featureVectors, targetVectors, initialTransient,
                              inputConnectivity=0.7, reservoirConnectivity=0.1, inputScaling=0.5,
                              reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3, learningMethod=LearningMethod.Batch,
                              reservoirActivationFunction = act.LogisticFunction(),
                              outputActivationFunction = act.ReLU()):


        inputWeightMatrix = topology.RandomInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size, inputConnectivity=inputConnectivity).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()

        if(learningMethod == LearningMethod.Batch):

            network = esn.Reservoir(size=size,
                                    spectralRadius=spectralRadius,
                                    inputScaling=inputScaling,
                                    reservoirScaling=reservoirScaling,
                                    leakingRate=leakingRate,
                                    initialTransient=initialTransient,
                                    inputData=featureVectors,
                                    outputData=targetVectors,
                                    inputWeightRandom=inputWeightMatrix,
                                    reservoirWeightRandom=reservoirWeightMatrix,
                                    reservoirActivationFunction=reservoirActivationFunction,
                                    outputActivationFunction=outputActivationFunction)
        else:
             network = onlineESN.Reservoir(size=size,
                                    spectralRadius=spectralRadius,
                                    inputScaling=inputScaling,
                                    reservoirScaling=reservoirScaling,
                                    leakingRate=leakingRate,
                                    initialTransient=initialTransient,
                                    inputData=featureVectors,
                                    outputData=targetVectors,
                                    inputWeightRandom=inputWeightMatrix,
                                    reservoirWeightRandom=reservoirWeightMatrix,
                                    reservoirActivationFunction=reservoirActivationFunction,
                                    outputActivationFunction=outputActivationFunction,
                                    batchLearnRatio=0.99)

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Store it and it will be used in the predictFuture method
        self.esn = network

    def getCorrelationMatrix(self, featureVectors, targetVectors, reservoirSize):

        # Ignore the bias
        inputSize = featureVectors.shape[1] -1
        correlationCoefficient = self.getCorrelationCoefficients(featureVectors[:, 1:], targetVectors)

        correlationMatrix = [np.random.rand(1,reservoirSize)]
        for i in range(inputSize):
            correlation = correlationCoefficient[0,i]
            correlationMatrix.append(np.ones((1, reservoirSize)) * correlation)
        correlationMatrix = np.array(correlationMatrix).reshape((inputSize+1, reservoirSize)).T
        return correlationMatrix

    def getRawCorrelationCoefficients(self, featureVectors, targetVectors):
        correlations = []
        y = targetVectors[:, 0]
        # For each feature vector, calculate the correlation coefficient
        for i in range(featureVectors.shape[1]):
            x = featureVectors[:, i]
            correlation, p_value = spearmanr(x,y)
            correlations.append(correlation)

        correlations = np.array(correlations)
        return correlations

    def getScaleCorrelationMatrix(self, correlations, scaling, inputSize, reservoirSize):

        # Scale the correlation coefficients
        inputSize = inputSize - 1

        if(scaling != 0.0):
            scaler = pp.MinMaxScaler((-scaling,scaling))
            correlations = scaler.fit_transform(correlations)
        correlations = correlations.reshape((1, inputSize))

        correlationMatrix = [np.random.rand(1,reservoirSize)]
        for i in range(inputSize):
            correlation = correlations[0,i]
            correlationMatrix.append(np.ones((1, reservoirSize)) * correlation)
        correlationMatrix = np.array(correlationMatrix).reshape((inputSize+1, reservoirSize)).T
        return correlationMatrix



    def trainESNWithoutTuningCorrelated(self, size, featureVectors, targetVectors, initialTransient,
                              inputConnectivity=0.7, reservoirConnectivity=0.1, inputScaling=0.5,
                              reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.2,
                              reservoirActivationFunction = act.LogisticFunction(),
                              outputActivationFunction = act.ReLU(),
                              learningMethod=LearningMethod.Batch,
                              correlatedScaling = 1.0):

        # Here, instead of assigning random input weights, assign the correlation as weights
        inputConnMatrix = topology.RandomInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size, inputConnectivity=inputConnectivity).generateConnectivityMatrix()
        correlationMatrix = self.getCorrelationMatrix(featureVectors, targetVectors, size)
        inputWeightMatrix = inputConnMatrix * correlationMatrix

        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()

        #reservoirWeightMatrix = topology.SmallWorldGraphs(size=size, meanDegree=int(size/2), beta=0.3).generateWeightMatrix()

        #reservoirWeightMatrix = topology.ScaleFreeNetworks(size=size, attachmentCount=int(size/2)).generateWeightMatrix()


        if(learningMethod == LearningMethod.Batch):

            network = esn.Reservoir(size=size,
                                    spectralRadius=spectralRadius,
                                    inputScaling=inputScaling,
                                    reservoirScaling=reservoirScaling,
                                    leakingRate=leakingRate,
                                    initialTransient=initialTransient,
                                    inputData=featureVectors,
                                    outputData=targetVectors,
                                    inputWeightRandom=inputWeightMatrix,
                                    reservoirWeightRandom=reservoirWeightMatrix,
                                    reservoirActivationFunction=reservoirActivationFunction,
                                    outputActivationFunction=outputActivationFunction)
        else:
             network = onlineESN.Reservoir(size=size,
                                    spectralRadius=spectralRadius,
                                    inputScaling=inputScaling,
                                    reservoirScaling=reservoirScaling,
                                    leakingRate=leakingRate,
                                    initialTransient=initialTransient,
                                    inputData=featureVectors,
                                    outputData=targetVectors,
                                    inputWeightRandom=inputWeightMatrix,
                                    reservoirWeightRandom=reservoirWeightMatrix,
                                    reservoirActivationFunction=reservoirActivationFunction,
                                    outputActivationFunction=outputActivationFunction,
                                    batchLearnRatio=0.99)

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Store it and it will be used in the predictFuture method
        self.esn = network


    def tuneLeakingRate(self, size, featureVectors, targetVectors, trainingSeries, validationSeries, initialTransient, depth, inputWeightMatrix, reservoirWeightMatrix,
                        reservoirActivationFunction, outputActivationFunction,
                        inputScaling=0.0, reservoirScaling=0.0, spectralRadius=0.79):

        # Now, tune the leaking rate to the optimal
        bestLeakingRate = None
        bestError = np.inf
        leakingRateList = np.arange(0.1, 1.0, 0.1)
        horizon = validationSeries.values.shape[0]


        for leakingRate in leakingRateList:
            # Create the reservoir
            res = esn.Reservoir(size=size,
                                spectralRadius=spectralRadius,
                                inputScaling=inputScaling,
                                reservoirScaling=reservoirScaling,
                                leakingRate=leakingRate,
                                initialTransient=initialTransient,
                                inputData=featureVectors,
                                outputData=targetVectors,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix,
                                reservoirActivationFunction=reservoirActivationFunction,
                                outputActivationFunction=outputActivationFunction)

            # Train the reservoir
            res.trainReservoir()

            # Warm up
            predictedTrainingOutputData = res.predict(featureVectors[-initialTransient:])

            # Predict for the validation data
            predictedSeries = self.predict(res, trainingSeries, depth, horizon)

            gc.collect()

            # Calcuate the regression error
            errorFunction = metrics.RootMeanSquareError()
            error = errorFunction.compute(validationSeries.values, predictedSeries.values)

            print("Leaking rate: "+str(leakingRate)+" Error: "+str(error))

            if(error < bestError):
                bestError = error
                bestLeakingRate = leakingRate

        return bestLeakingRate


    def trainESNWithCorrelationTuned(self, size, trainingSeries, validationSeries, availableSeries,
                                     initialTransient, depth, inputConnectivity=0.7, reservoirConnectivity=0.3,
                                     inputScaling=0.0,
                              reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3,
                              reservoirActivationFunction = act.LogisticFunction(),
                              outputActivationFunction = act.ReLU()):

        # Form the feature and target vectors
        trainingInputData, trainingOutputData = self.formContinousFeatureAndTargetVectors(trainingSeries, depth)
        correlationCoefficients = self.getRawCorrelationCoefficients(trainingInputData[:, 1:], trainingOutputData)
        correlationMatrix = self.getScaleCorrelationMatrix(correlationCoefficients, scaling=0.0, inputSize=trainingInputData.shape[1], reservoirSize=size)

        # Input Conn Matrix and Reservoir weight matrix
        inputConnMatrix = topology.RandomInputTopology(inputSize=trainingInputData.shape[1], reservoirSize=size, inputConnectivity=inputConnectivity).generateConnectivityMatrix()
        inputWeightMatrix = inputConnMatrix * correlationMatrix
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()

        print("Tuning the leaking rate..")

        # # Tune the leaking rate
        # bestLeakingRate = self.tuneLeakingRate(size=size, featureVectors=trainingInputData, targetVectors=trainingOutputData, trainingSeries=trainingSeries, validationSeries=validationSeries,
        #                                        initialTransient=initialTransient, depth=depth, inputWeightMatrix=inputWeightMatrix, reservoirWeightMatrix=reservoirWeightMatrix,
        #                                        reservoirActivationFunction=reservoirActivationFunction, outputActivationFunction=outputActivationFunction)
        #
        # print("Optimal leaking rate:"+str(bestLeakingRate))
        #
        # print("Tuning the input scaling..")

        correlationTuner = CorrelationBruteTuner(size=size,
                                                 initialTransient=initialTransient,
                                                 trainingSeries=trainingSeries,
                                                 validationSeries=validationSeries,
                                                 arbitraryDepth=depth,
                                                 reservoirActivationFunction = reservoirActivationFunction,
                                                 outputActivationFunction=outputActivationFunction,
                                                 featureVectors=trainingInputData,
                                                 targetVectors=trainingOutputData,
                                                 correlationCoefficients=correlationCoefficients,
                                                 inputConnMatrix=inputConnMatrix,
                                                 reservoirWeightMatrix=reservoirWeightMatrix
                                                 )

        correlationScalingOptimum, leakingRateOptimum = correlationTuner.getOptimalParameters()
        print("Optimal leaking rate.."+str(leakingRateOptimum))
        print("Optimal scaling.."+str(correlationScalingOptimum))

        # Input weight matrix
        correlationMatrix = self.getScaleCorrelationMatrix(correlationCoefficients, correlationScalingOptimum,
                                                           trainingInputData.shape[1], size)
        inputWeightMatrix = inputConnMatrix * correlationMatrix



        # Train and save the network with optimal parameters
        network = esn.Reservoir(size=size,
                                spectralRadius=spectralRadius,
                                inputScaling=inputScaling,
                                reservoirScaling=reservoirScaling,
                                leakingRate=leakingRateOptimum,
                                initialTransient=initialTransient,
                                inputData=trainingInputData,
                                outputData=trainingOutputData,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix,
                                reservoirActivationFunction=reservoirActivationFunction,
                                outputActivationFunction=outputActivationFunction)


        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(trainingInputData[-initialTransient:])

        # Store it and it will be used in the predictFuture method
        self.esn = network


    def trainESNWithTuning(self, size, featureVectors, targetVectors, initialTransient,
                       initialSeedSeries, validationOutputData, arbitraryDepth, featureIndices,
                       inputConnectivity=1.0, reservoirConnectivity=0.5):

        inputWeightMatrix = topology.RandomInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size, inputConnectivity=inputConnectivity).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()


        resTuner = ReservoirParameterTuner(size=size, initialTransient=initialTransient,
                                           trainingInputData=featureVectors, trainingOutputData=targetVectors,
                                           initialSeedSeries=initialSeedSeries, validationOutputData=validationOutputData,
                                           arbitraryDepth= arbitraryDepth, featureIndices=featureIndices,
                                           spectralRadiusBound=(0.0,1.0), inputScalingBound=(0.0,1.0),
                                           reservoirScalingBound=(0.0,1.0), leakingRateBound=(0.0,1.0),
                                           inputWeightMatrix=inputWeightMatrix, reservoirWeightMatrix=reservoirWeightMatrix,
                                           minimizer=Minimizer.DifferentialEvolution)
        spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum = resTuner.getOptimalParameters()


        network = esn.Reservoir(size=size,
                                spectralRadius=spectralRadiusOptimum,
                                inputScaling=inputScalingOptimum,
                                reservoirScaling=reservoirScalingOptimum,
                                leakingRate=leakingRateOptimum,
                                initialTransient=initialTransient,
                                inputData=featureVectors,
                                outputData=targetVectors,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix,
                                activationFunction=esn.ActivationFunction.EXPIT,
                                outputRelu=True)

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Store it and it will be used in the predictFuture method
        self.esn = network


    def predict(self, network, availableSeries, arbitraryDepth, horizon):
        # To avoid mutation of pandas series
        initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
        for i in range(horizon):
            feature = initialSeries.values[-arbitraryDepth:].reshape((1, arbitraryDepth))

            # Append bias
            feature = np.hstack((1.0,feature[0, :])).reshape((1, feature.shape[1]+1))

            nextPoint = network.predictOnePoint(feature)[0]

            nextIndex = initialSeries.last_valid_index() + pd.Timedelta(hours=1)
            initialSeries[nextIndex] = nextPoint

        predictedSeries = initialSeries[-horizon:]
        return predictedSeries

    def predictFutureDays(self, network, availableSeries, arbitraryDepth, horizon):
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



    def predictFuture(self, availableSeries, depth, horizon):

        # future series
        futureSeries = pd.Series()

        # Feature list
        featureIntervalList = []
        for i in range(depth, 0, -1):
            interval = pd.Timedelta(hours=-(i))
            featureIntervalList.append(interval)

        nextValue = availableSeries.last_valid_index()
        for i in range(horizon):
            nextValue = nextValue + pd.Timedelta(hours=1)

            #Form the feature vectors
            feature = [1.0]
            #feature = []
            for interval in featureIntervalList:
                feature.append(availableSeries[nextValue + interval])

            feature = np.array(feature).reshape((1,len(featureIntervalList)+1))

            predictedValue = self.esn.predictOnePoint(feature)[0]

            # Add to the future list
            futureSeries[nextValue] = predictedValue

            # Add it to the series
            availableSeries[nextValue] = predictedValue

        return futureSeries

    def predictFutureFeatureTransformer(self, esn, availableSeries, depth, horizon):

        # future series
        futureSeries = pd.Series()

        # Feature list
        featureIntervalList = []
        for i in range(depth, 0, -1):
            interval = pd.Timedelta(hours=-(i))
            featureIntervalList.append(interval)

        nextValue = availableSeries.last_valid_index()
        for i in range(horizon):
            nextValue = nextValue + pd.Timedelta(hours=1)

            #Form the feature vectors
            #feature = [1.0]
            feature = []
            for interval in featureIntervalList:
                feature.append(availableSeries[nextValue + interval])

            feature = np.array(feature).reshape((1,len(featureIntervalList)))

            predictedValue = esn.predictOnePoint(feature)[0,0]

            # Add to the future list
            futureSeries[nextValue] = predictedValue

            # Add it to the series
            availableSeries[nextValue] = predictedValue

        return futureSeries

    def predictFutureWithPCA(self, availableSeries, depth, horizon, pca):

        # future series
        futureSeries = pd.Series()

        # Feature list
        featureIntervalList = []
        for i in range(depth, 0, -1):
            interval = pd.Timedelta(hours=-(i))
            featureIntervalList.append(interval)

        nextValue = availableSeries.last_valid_index()
        for i in range(horizon):
            nextValue = nextValue + pd.Timedelta(hours=1)

            # Form the feature vectors
            feature = []
            for interval in featureIntervalList:
                feature.append(availableSeries[nextValue + interval])

            feature = np.array(feature).reshape((1,len(featureIntervalList)))

            # Transform the feature vector into PCA reduced dimensions
            feature = pca.transform(feature)
            featureLen = feature.shape[1]

            # Add the bias
            feature = np.hstack((1.0,feature[0, :])).reshape((1, featureLen+1))

            predictedValue = self.esn.predict(feature)[0,0]

            # Add to the future list
            futureSeries[nextValue] = predictedValue

            # Add it to the series
            availableSeries[nextValue] = predictedValue

        return futureSeries


    def predictFutureWithFeatureInterval(self, availableSeries, featureIntervalList, horizon):

        # future series
        futureSeries = pd.Series()

        nextValue = availableSeries.last_valid_index()
        for i in range(horizon):
            nextValue = nextValue + pd.Timedelta(hours=1)

            # Form the feature vectors
            feature = [1.0]
            for interval in featureIntervalList:
                feature.append(availableSeries[nextValue + interval])

            feature = np.array(feature).reshape((1,len(featureIntervalList)+1))

            predictedValue = self.esn.predict(feature)[0,0]

            # Add to the future list
            futureSeries[nextValue] = predictedValue

            # Add it to the series
            availableSeries[nextValue] = predictedValue

        return futureSeries


    def plotSeries(self, folderName, seriesList, seriesNameList, title, subTitle, fileName="Prediction.html"):
        os.mkdir(folderName)
        outplot = plotting.OutputTimeSeriesPlot(folderName + "/" + fileName, title, "", subTitle)

        for i in range(len(seriesList)):
            series = seriesList[i]
            xAxis = []
            for index, value in series.iteritems():
                year = index.strftime("%Y")
                month = index.strftime("%m")
                day = index.strftime("%d")
                hour = index.strftime("%H")
                nextDayStr = "Date.UTC(" + str(year)+","+ str(int(month)-1) + "," + str(day) + ","+ str(hour)+")"
                xAxis.append(nextDayStr)

            outplot.setSeries(seriesNameList[i], np.array(xAxis), series.values)
        outplot.createOutput()

    def plotCombinedSeries(self, folderName, seriesList, seriesNameList, title, subTitle, yAxisText, fileName="Prediction.html"):
        os.mkdir(folderName)
        outplot = combPlotting.TimeSeriesLineBarPlot(folderName + "/" + fileName, title, subTitle, yAxisText)

        for i in range(len(seriesList)):
            series = seriesList[i]
            xAxis = []
            for index, value in series.iteritems():
                year = index.strftime("%Y")
                month = index.strftime("%m")
                day = index.strftime("%d")
                hour = index.strftime("%H")
                nextDayStr = "Date.UTC(" + str(year)+","+ str(int(month)-1) + "," + str(day) + ","+ str(hour)+")"
                xAxis.append(nextDayStr)

            outplot.setSeries(seriesNameList[i], np.array(xAxis), series.values)
        outplot.createOutput()

    def filterRecent(self, series, count):
        data = series.values[-count:]
        index = series.index[-count:]
        return pd.Series(data=data, index=index)

    def getFeatures(self, series, dropThreshold):

        # Get the feature and target vectors to an arbitrary depth
        depth = 24 * 60 # 1 year
        featureVectors, targetVectors = self.formContinousFeatureAndTargetVectorsWithoutBias(series, depth)

        features = []
        y = targetVectors[:, 0]
        correlations = []
        # For each feature vector, calculate the correlation coefficient
        for i in range(featureVectors.shape[1]):
            x = featureVectors[:, i]
            correlation, p_value = pearsonr(x,y)
            correlations.append(abs(correlation))

        # Scale the correlations
        correlations = np.array(correlations)
        scaler = pp.MinMaxScaler((0,1))
        correlations = scaler.fit_transform(correlations)


        # Get the best ones
        for i in range(correlations.shape[0]):
            if(correlations[i] > dropThreshold):
                featureDepth = depth - i
                features.append(pd.Timedelta(hours=-featureDepth))
        return features

    def getCorrelationCoefficients(self, featureVectors, targetVectors):
        correlations = []
        y = targetVectors[:, 0]
        # For each feature vector, calculate the correlation coefficient
        for i in range(featureVectors.shape[1]):
            x = featureVectors[:, i]
            #correlation, p_value = pearsonr(x,y)
            correlation, p_value = spearmanr(x,y)
            correlations.append(correlation)

        # # Scale the correlations
        correlations = np.array(correlations)

        # Normalize - Find the sum and divide by it
        # sum = np.sum(correlations)
        # correlations = correlations / sum

        sum = np.sum(correlations)
        correlations = correlations / sum

        #scaler = pp.MinMaxScaler((-1,1))
        #correlations = scaler.fit_transform(correlations)

        #norm = np.linalg.norm(correlations)
        #correlations = correlations / norm



        # Re-shape
        correlations = correlations.reshape((1, featureVectors.shape[1]))
        return correlations

    def getFeaturesForThreshold(self, featureVectors, correlationCoefficients, cutoffThreshold):
        featureIndices = np.where(correlationCoefficients.flatten() >= cutoffThreshold)[0]

        # Select the features using the indices
        features = featureVectors[:, featureIndices]

        return features, featureIndices


    def trainAndPredict(self, availableSeries, featureIndices, featureVectors, targetVectors, arbitraryDepth, horizon, networkParameters):
        size = networkParameters['size']
        initialTransient = networkParameters['initialTransient']
        spectralRadius = networkParameters['spectralRadius']
        inputScaling = networkParameters['inputScaling']
        reservoirScaling = networkParameters['reservoirScaling']
        leakingRate = networkParameters['leakingRate']
        inputConnectivity = networkParameters['inputConnectivity']
        reservoirConnectivity = networkParameters['reservoirConnectivity']

        inputWeightMatrix = topology.RandomInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size, inputConnectivity=inputConnectivity).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()

        network = esn.Reservoir(size=size,
                                spectralRadius=spectralRadius,
                                inputScaling=inputScaling,
                                reservoirScaling=reservoirScaling,
                                leakingRate=leakingRate,
                                initialTransient=initialTransient,
                                inputData=featureVectors,
                                outputData=targetVectors,
                                inputWeightRandom=inputWeightMatrix,
                                reservoirWeightRandom=reservoirWeightMatrix,
                                reservoirActivationFunction=act.LogisticFunction(),
                                outputActivationFunction=act.ReLU())

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Start predicting horizon number of points
        predictedSeries = self.predict(network, availableSeries, arbitraryDepth, horizon, featureIndices)
        return predictedSeries

    def predictI(self, network, availableSeries, arbitraryDepth, horizon, featureIndices):
        # To avoid mutation of pandas series
        initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
        for i in range(horizon):
            feature = initialSeries.values[-arbitraryDepth:].reshape((1, arbitraryDepth))
            feature = feature[:, featureIndices]

            #Append bias
            feature = np.hstack((1.0,feature[0, :])).reshape((1, feature.shape[1]+1))

            nextPoint = network.predictOnePoint(feature)[0]

            nextIndex = initialSeries.last_valid_index() + pd.Timedelta(hours=1)
            initialSeries[nextIndex] = nextPoint

        predictedSeries = initialSeries[-horizon:]
        return predictedSeries

    def predictHierarchy(self, network, availableSeries, arbitraryDepth, horizon, featureIndices):
        # To avoid mutation of pandas series
        initialSeries = pd.Series(data=availableSeries.values, index=availableSeries.index)
        for i in range(horizon):
            feature = initialSeries.values[-arbitraryDepth:].reshape((1, arbitraryDepth))
            #feature = feature[:, featureIndices]

            #Append bias
            #feature = np.hstack((1.0,feature[0, :])).reshape((1, feature.shape[1]+1))

            nextPoint = network.predictOnePoint(feature)[0]

            nextIndex = initialSeries.last_valid_index() + pd.Timedelta(hours=1)
            initialSeries[nextIndex] = nextPoint

        predictedSeries = initialSeries[-horizon:]
        return predictedSeries

    def getBestLeakingRate(self,featureIndices, depth, featureVectors, targetVectors, availableSeries, validationSeries, networkParamaters):
        bestError = np.inf
        bestLeakingRate = None
        leakingRateRange = np.arange(0.1, 1.0, 0.1).tolist()
        errorFun = metrics.MeanSquareError()
        horizon = validationSeries.values.shape[0]

        for leakingRate in leakingRateRange:
            # Train the network and get the predicted Series
            networkParamaters['leakingRate'] = leakingRate
            predictedSeries = self.trainAndPredict(availableSeries, featureIndices, featureVectors, targetVectors, depth, horizon, networkParamaters)

            # Measure the error between predicted series and validation series
            validationError = errorFun.compute(validationSeries.values, predictedSeries.values)

            print("Leaking Rate: "+str(leakingRate)+" Regression Error: "+ str(validationError))

            if(validationError < bestError):
                bestError = validationError
                bestLeakingRate = leakingRate
        return bestLeakingRate


    def getBestFeatures(self, trainingSeries, validationSeries, networkParameters, method, args={}):

        # # # Form the feature vectors to an arbitrary large depth
        horizon = validationSeries.values.shape[0]
        # # featureVectors, targetVectors = self.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, maxDepth)
        #
        # # Calculate the correlation coefficients
        # correlationCoefficients = self.getCorrelationCoefficients(featureVectors, targetVectors)

        if(method == FeatureSelectionMethod.CutOff_Threshold):
            # Now, vary the cut-off threshold and get the features and choose the best one
            # TODO: Also, have to think about the depth (Probably, this has to be tuned as well)
            errorFun = metrics.MeanSquareError()
            thresholdRange = np.arange(0.1, 0.6, 0.2).tolist()
            bestIndices = None
            bestFeatures = None
            bestError = np.inf
            for i in thresholdRange:
                features, indices = self.getFeaturesForThreshold(featureVectors, correlationCoefficients, i)
                features = np.hstack((np.ones((features.shape[0], 1)), features))

                # Train the network and get the predicted Series
                predictedSeries = self.trainAndPredict(trainingSeries, indices, features, targetVectors, depth, horizon, networkParameters)

                # Measure the error between predicted series and validation series
                validationError = errorFun.compute(validationSeries.values, predictedSeries.values)

                print("Cut-off threshold: "+str(i)+" Regression Error: "+ str(validationError))

                if(validationError < bestError):
                    bestError = validationError
                    bestIndices = np.copy(indices) # Stupid mutations TODO: Check for mutation in other places
                    bestFeatures = np.copy(features)

            # Return the best features, indices, and target vectors
            return bestIndices, bestFeatures, targetVectors

        elif(method == FeatureSelectionMethod.Pattern_Analysis):
            bestDepth = None
            bestIndices = None
            bestFeatures = None
            bestTarget = None
            bestError = np.inf
            bestR2score = -1.0
            depthRange = np.arange(30*24, 120*24, 30*24).tolist()
            for depth in depthRange:
                featureVectors, targetVectors = self.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, depth)

                # Calculate the correlation coefficients
                correlationCoefficients = self.getCorrelationCoefficients(featureVectors, targetVectors)

                # Get the empty feature correlation bins
                featureCorrelationBins = self.getEmptyFeatureCorrelationBins()

                # Collect the correlations in their respective bins
                featureCorrelationBins = self.formFeatureCorrelationBins(featureCorrelationBins, correlationCoefficients)

                # Sort them
                sortedBins = sorted(featureCorrelationBins.items(), key=operator.itemgetter(1), reverse=True)

                # Now, select the good features, this is done using dropping features with less correlation coefficient
                errorFun = metrics.MeanSquareError()
                #errorFun = metrics.R2Score()
                thresholdRange = np.arange(0.0, 1.0, 0.2).tolist()
                for threshold in thresholdRange:
                    # Pick the bins which has average correlation coefficient greater than the cutoff threshold
                    bestBins = [ bin for bin in sortedBins if bin[1][0] >= threshold]

                    # Stop when there are no feature bins
                    if(len(bestBins) == 0):
                        break

                    # Form the features
                    indices = self.getFeatureIndices(bestBins)
                    features = featureVectors[:, indices]
                    features = np.hstack((np.ones((features.shape[0], 1)), features))

                    # Train the network and get the predicted Series
                    predictedSeries = self.trainAndPredict(trainingSeries, indices, features, targetVectors, depth, horizon, networkParameters)

                    # Measure the error between predicted series and validation series
                    validationError = errorFun.compute(validationSeries.values, predictedSeries.values)
                    print("Cut-off threshold: "+str(threshold)+ " Depth: "+str(depth/24)+" Regression Error: "+ str(validationError))

                    # # Measure the R2 score
                    # r2score = errorFun.compute(validationSeries.values, predictedSeries.values)
                    # print("Cut-off threshold: "+str(threshold)+" R2 score: "+ str(r2score))

                    # if(r2score > bestR2score):
                    #     bestR2score = r2score
                    #     bestIndices = np.copy(indices)
                    #     bestFeatures = np.copy(features)

                    if(validationError < bestError):
                        bestDepth = depth
                        bestError = validationError
                        bestIndices = np.copy(indices) # Stupid mutations TODO: Check for mutation in other places
                        bestFeatures = np.copy(features)
                        bestTarget = np.copy(targetVectors)

            # Return the best features, indices, and target vectors
            print(bestIndices)
            return bestDepth, bestIndices, bestFeatures, bestTarget

    def getFeatureIndices(self, bins):

        featureIndices = []

        for bin in bins:
            indices = bin[1][1]
            featureIndices.extend(indices)

        # Get the unique
        featureIndices = list(set(featureIndices))

        return featureIndices

    def getEmptyFeatureCorrelationBins(self):
        featureBins = {}  # Dict of depth and correlation
        # Key is a tuple containing depth, isInterval, delta
        # Initialize all the bins with cumulative correlation coefficient of zero
        for i in range(1,11):  # (t-1),(t-2).....(t-11)
            featureBins[(i, False, 0)] = 0.0, []

        # Interval in terms of bi-daily, daily, and weekly
        maxDelta = 3
        for interval in [12, 24]:
            featureBins[(interval, True, 0)] = 0.0, []
            for delta in range(1,maxDelta+1): # (t-10),(t-11)(t-interval),(t-13),(t-14)
                featureBins[(interval, True, -delta)] = 0.0, []
                featureBins[(interval, True, +delta)] = 0.0, []
        return featureBins

    def formFeatureCorrelationBins(self, bins, correlationCoefficients):

        totalSize = correlationCoefficients.shape[1]
        for i in range(totalSize):
            depthIndex = totalSize - i
            correlation = correlationCoefficients[0, i]

            if(depthIndex in range(1,11)):
                newCorrelation = bins[(depthIndex, False, 0)][0] + correlation
                newIndices = bins[(depthIndex, False, 0)][1]
                newIndices.append(i)
                bins[(depthIndex, False, 0)] = newCorrelation, newIndices

            maxDelta = 3
            for interval in [12,24]:
                if(depthIndex >= interval):
                    if(depthIndex%interval == 0):
                            newCorrelation = bins[(interval, True, 0)][0] + correlation
                            newIndices = bins[(interval, True, 0)][1]
                            newIndices.append(i)
                            bins[(interval, True, 0)] = newCorrelation, newIndices
                    for delta in range(1, maxDelta+1):
                        if((depthIndex-delta)%interval == 0):
                            newCorrelation = bins[(interval, True, delta)][0] + correlation
                            newIndices = bins[(interval, True, delta)][1]
                            newIndices.append(i)
                            bins[(interval, True, delta)] = newCorrelation, newIndices
                        if((depthIndex+delta)%interval == 0):
                            newCorrelation = (bins[(interval, True, -delta)][0] + correlation)/2
                            newIndices = bins[(interval, True, -delta)][1]
                            newIndices.append(i)
                            bins[(interval, True, -delta)] = newCorrelation, newIndices

        # Calculate the average
        for key, value in bins.items():
            correlation, indices = value
            correlation = correlation/len(indices)
            bins[key] = correlation, indices

        return bins

if __name__ == '__main__':
    util= SeriesUtility()
    ccBins = util.getEmptyFeatureCorrelationBins()





