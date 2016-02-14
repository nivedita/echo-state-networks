import pandas as pd
import numpy as np
from timeseries import TimeSeriesContinuousProcessor as processor, TimeSeriesInterval as tsi
from reservoir import classicESN as esn, ReservoirTopology as topology
from sklearn import preprocessing as pp
import os
from plotting import OutputTimeSeries as plotting
# from reservoir import Tuner as tuner
from performance import ErrorMetrics as metrics
from enum import Enum
import gc
from scipy import optimize
from scipy.stats import pearsonr

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
                 initialSeedSeries, depth, validationOutputData, spectralRadiusBound, inputScalingBound,
                 reservoirScalingBound, leakingRateBound,inputWeightMatrix=None,
                 reservoirWeightMatrix=None, minimizer=Minimizer.DifferentialEvolution,
                 initialGuess = np.array([0.79, 0.5, 0.5, 0.3])):
        self.size = size
        self.initialTransient = initialTransient
        self.trainingInputData = trainingInputData
        self.trainingOutputData = trainingOutputData
        self.initialSeedSeries = initialSeedSeries
        self.depth = depth
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
                                  activationFunction=esn.ActivationFunction.EXPIT)

        #Train the reservoir
        res.trainReservoir()

        # Warm up
        predictedTrainingOutputData = res.predict(self.trainingInputData[-self.initialTransient:])

        #Predict for the validation data
        predictedOutputData = self.predictFromInitialSeed(res, self.initialSeedSeries, self.depth, self.horizon)

        gc.collect()

        #Calcuate the regression error
        errorFunction = metrics.MeanSquareError()
        regressionError = errorFunction.compute(self.validationOutputData, predictedOutputData)

        #Return the error
        print("\nThe Parameters: "+str(x)+" Regression error:"+str(regressionError))
        return regressionError

    def predictFromInitialSeed(self, res, initialSeedSeries, depth, horizon):
        availableSeries = initialSeedSeries

        for i in range(horizon):
            feature = availableSeries.values[-depth:]
            feature = np.swapaxes(np.hstack((1.0, feature)).reshape((depth+1,1)),1,0)

            nextPoint = res.predictOnePoint(feature)[0]

            nextIndex = availableSeries.last_valid_index() + pd.Timedelta(hours=1)
            availableSeries[nextIndex] = nextPoint

        predictedOutputData = availableSeries.values[-horizon:]
        return predictedOutputData


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

    def _mean(self, x):
        if len(x) == 0:
            return 0
        else:
            return sum(x)/len(x)

    def resampleSeriesExists(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._exists)

    def resampleSeriesSum(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._sum)

    def resampleSeriesMean(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._mean)

    def scaleSeries(self, series):
        self.scalingFunction = pp.MinMaxScaler((0,1))
        #self.scalingFunction = pp.StandardScaler()
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
                              inputConnectivity=0.8, reservoirConnectivity=0.4, inputScaling=0.5,
                              reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3):


        inputWeightMatrix = topology.ClassicInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()
        #reservoirWeightMatrix = topology.ScaleFreeNetworks(size=size, attachmentCount=int(size/2)).generateWeightMatrix()
        #reservoirWeightMatrix = topology.SmallWorldGraphs(size=size, meanDegree=int(size/2), beta=0.5).generateWeightMatrix()


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
                                activationFunction=esn.ActivationFunction.EXPIT,
                                outputRelu=True)

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Store it and it will be used in the predictFuture method
        self.esn = network

    def trainESNWithTuning(self, size, featureVectors, targetVectors, initialTransient,
                       initialSeedSeries, depth, validationOutputData,
                       inputConnectivity=1.0, reservoirConnectivity=0.5, inputScaling=0.5,
                       reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.2):


        inputWeightMatrix = topology.ClassicInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size).generateWeightMatrix()
        reservoirWeightMatrix = topology.RandomReservoirTopology(size=size, connectivity=reservoirConnectivity).generateWeightMatrix()


        resTuner = ReservoirParameterTuner(size=size, initialTransient=initialTransient,
                                           trainingInputData=featureVectors, trainingOutputData=targetVectors,
                                           initialSeedSeries=initialSeedSeries, depth=depth, validationOutputData=validationOutputData,
                                           spectralRadiusBound=(0.0,1.0), inputScalingBound=(0.0,1.0),
                                           reservoirScalingBound=(0.0,1.0), leakingRateBound=(0.0,1.0),
                                           inputWeightMatrix=inputWeightMatrix, reservoirWeightMatrix=reservoirWeightMatrix,
                                           minimizer=Minimizer.BasinHopping)
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
                                activationFunction=esn.ActivationFunction.EXPIT)

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Store it and it will be used in the predictFuture method
        self.esn = network



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
            for interval in featureIntervalList:
                feature.append(availableSeries[nextValue + interval])

            feature = np.array(feature).reshape((1,len(featureIntervalList)+1))

            predictedValue = self.esn.predict(feature)[0,0]

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
            correlation, p_value = pearsonr(x,y)
            correlations.append(abs(correlation))

        # Scale the correlations
        correlations = np.array(correlations)
        scaler = pp.MinMaxScaler((0,1))
        correlations = scaler.fit_transform(correlations)

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
        reservoirConnectivity = networkParameters['reservoirConnectivity']

        inputWeightMatrix = topology.ClassicInputTopology(inputSize=featureVectors.shape[1], reservoirSize=size).generateWeightMatrix()
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
                                activationFunction=esn.ActivationFunction.EXPIT,
                                outputRelu=True)

        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors[-initialTransient:])

        # Start predicting horizon number of points
        predictedSeries = self.predict(network, availableSeries, arbitraryDepth, horizon, featureIndices)
        return predictedSeries

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


    def getBestFeatures(self, trainingSeries, validationSeries, networkParameters):

        # Form the feature vectors to an arbitrary large depth
        depth = networkParameters['arbitraryDepth']
        horizon = validationSeries.values.shape[0]
        featureVectors, targetVectors = self.formContinousFeatureAndTargetVectorsWithoutBias(trainingSeries, depth)

        # Calculate the correlation coefficients
        correlationCoefficients = self.getCorrelationCoefficients(featureVectors, targetVectors)

        # Now, vary the cut-off threshold and get the features and choose the best one
        # TODO: Also, have to think about the depth (Probably, this has to be tuned as well)

        errorFun = metrics.MeanSquareError()
        thresholdRange = np.arange(0.1, 0.5, 0.02).tolist()
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






