import pandas as pd
import numpy as np

class SeriesUtility:

    'Convert csv files to pandas time series'
    def convertDatasetsToSeries(self, fileName):
        # Read the data
        df = pd.read_csv(fileName, index_col=0, parse_dates=True, names=['value'])

        # Convert the dataframe into series
        data = np.array(df.as_matrix()).flatten().astype(float)
        series = pd.Series(data=data, index=df.index)
        return series

    'Unify multiple series such that the start and end are synced'
    def unifyMultipleSeries(self, seriesList):

    'Aggregate multiple series into one based on the aggregation rules like sum, mean etc'
    def aggregateSeries(self, seriesList, aggregationRule):

    'Plot the predicted series in comparison with actual series'
    def plotSeriesPrediction(self, seriesActual, seriesPredicted):

    def _sum(x):
        if len(x) == 0:
            return 0
        else:
            return sum(x)

    'Resample the series based on the sampleing rule like 1H, 4D etc'
    def resampleSeries(self, series, samplingRule):
        return series.resample(rule=samplingRule, how=self._sum)

    'Scale the series as per the passed scaling function like MinMax Scaler'
    def scaleSeries(self, series, scalingFunction):
        data = scalingFunction.fit_transform(series.values)
        index = series.index
        scaledSeries = pd.Series(data=data,index=index)
        return scaledSeries

    def splitIntoTrainingAndTestingSeries(self, series, horizon):
        index = series.index
        data = series.data
        trainingData = data[:data.shape[0]-horizon]
        trainingIndex = index[:index.shape[0]-horizon]
        testingData = data[data.shape[0]-horizon:]
        testingIndex = index[index.shape[0]-horizon:]

        trainingSeries = pd.Series(data=trainingData.flatten(),index=trainingIndex)
        testingSeries = pd.Series(data=testingData.flatten(),index=testingIndex)
        return trainingSeries, testingSeries

