import pandas as pd
import numpy as np
from timeseries import TimeSeriesContinuousProcessor as processor, TimeSeriesInterval as tsi
from reservoir import EchoStateNetwork as esn, ReservoirTopology as topology
from sklearn import preprocessing as pp
import os
from plotting import OutputTimeSeries as plotting
from reservoir import Tuner as tuner

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

    def formContinousFeatureAndTargetVectors(self, series, depth):
        # Pre-process the data and form feature and target vectors
        tsp = processor.TimeSeriesContinuosProcessor(series, depth, horizon=1)
        featureVectors, targetVectors = tsp.getProcessedData()

        # Append bias to feature vectors
        featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)), featureVectors))

        return featureVectors, targetVectors

    def trainESNWithoutTuning(self, size, featureVectors, targetVectors, initialTransient,
                              inputConnectivity=0.8, reservoirConnectivity=0.4, inputScaling=0.5,
                              reservoirScaling=0.5, spectralRadius=0.79, leakingRate=0.3):

        network = esn.EchoStateNetwork(size=size,
                                       inputData=featureVectors,
                                       outputData=targetVectors,
                                       reservoirTopology=topology.RandomTopology(size=size,
                                                                                 connectivity=reservoirConnectivity),
                                       spectralRadius=spectralRadius,
                                       inputConnectivity=inputConnectivity,
                                       inputScaling=inputScaling,
                                       reservoirScaling=reservoirScaling,
                                       leakingRate=leakingRate)
        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors)

        # Store it and it will be used in the predictFuture method
        self.esn = network

    def trainESNWithFullTuning(self, size, featureVectors, targetVectors, initialTransient):

        inputConnectivityBound = (0.1, 1.0)
        reservoirConnectivityBound = (0.1, 1.0)
        inputScalingBound = (0.0, 1.0)
        reservoirScalingBound = (0.0, 1.0)
        spectralRadiusBound = (0.0, 1.0)
        leakingRateBound = (0.0, 1.0)

        esnTuner = tuner.ESNTunerWithConnectivity(size=size,
                                                  initialTransient=initialTransient,
                                                  trainingInputData=featureVectors,
                                                  trainingOutputData=targetVectors,
                                                  validationInputData=featureVectors,
                                                  validationOutputData=targetVectors,
                                                  spectralRadiusBound=spectralRadiusBound,
                                                  inputScalingBound=inputScalingBound,
                                                  reservoirScalingBound=reservoirScalingBound,
                                                  leakingRateBound=leakingRateBound,
                                                  inputConnectivityBound=inputConnectivityBound,
                                                  reservoirConnectivityBound=reservoirConnectivityBound)
        spectralRadiusOptimum, inputScalingOptimum, reservoirScalingOptimum, leakingRateOptimum, inputWeightConn, reservoirWeightConn = esnTuner.getOptimalParameters()

        network = esn.EchoStateNetwork(size=size,
                                       inputData=featureVectors,
                                       outputData=targetVectors,
                                       reservoirTopology=topology.RandomTopology(size=size, connectivity=0.3),
                                       spectralRadius=spectralRadiusOptimum,
                                       inputScaling=inputScalingOptimum,
                                       reservoirScaling=reservoirScalingOptimum,
                                       leakingRate=leakingRateOptimum,
                                       initialTransient=initialTransient,
                                       inputWeightConn=inputWeightConn,
                                       reservoirWeightConn=reservoirWeightConn)
        network.trainReservoir()

        # Warm-up the network
        trainingPredictedOutputData = network.predict(featureVectors)

        # Store it and it will be used in the predictFuture method
        self.esn = network

    def trainESNWithMinimalTuning(self, size, featureTrainingVectors, targetTrainingVectors,
                                  featureValidationVectors, targetValidationVectors, initialTransient):

        inputConnectivityBound = (0.1,0.9)
        probabilityBound = (0.1,0.9)

        esnTuner = tuner.ESNErdosTuner(size=300,
                                  initialTransient=initialTransient,
                                  trainingInputData=featureTrainingVectors,
                                  trainingOutputData=targetTrainingVectors,
                                  validationInputData=featureValidationVectors,
                                  validationOutputData=targetValidationVectors,
                                  inputConnectivityBound=inputConnectivityBound,
                                  probabilityBound=probabilityBound,
                                  times=3)
        inputConnectivityOptimum, probabilityOptimum = esnTuner.getOptimalParameters()

        network = esn.EchoStateNetwork(size=size,
                                       inputData=featureTrainingVectors,
                                       outputData=targetTrainingVectors,
                                       reservoirTopology=topology.ErdosRenyiTopology(size=size,
                                                                                     probability=probabilityOptimum),
                                       initialTransient=initialTransient,
                                       inputConnectivity=inputConnectivityOptimum)
        network.trainReservoir()

        # # Warm-up the network
        # trainingPredictedOutputData = network.predict(featureValidationVectors)

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
