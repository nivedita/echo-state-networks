import numpy as np
import pandas as pd
class TimeSeriesIntervalProcessor:

    def __init__(self, series, featureIntervalList, targetIntervalList):
        self.series = series
        self.targetIntervalList = targetIntervalList
        self.featureIntervalList = featureIntervalList

    def __generate__(self):

        featureVectors = []
        targetVectors = []

        # Now, for each index in indices, form the feature and target vector
        for index in self.series.index:

            feature = []
            target = []
            missingFeature = False
            missingTarget = False
            #Feature vector
            for interval in self.featureIntervalList:
                intervalIndex = index+interval
                if(intervalIndex in self.series.index):
                    valueAtInterval = self.series[intervalIndex]
                    feature.append(valueAtInterval)
                else:
                    missingFeature = True
                    break


            #Target vector
            for interval in self.targetIntervalList:
                intervalIndex = index+interval
                if(intervalIndex in self.series.index):
                    valueAtInterval = self.series[index+interval]
                    target.append(valueAtInterval)
                else:
                    missingTarget = False
                    break


            if(not missingFeature and not missingTarget):
                feature = np.array(feature).reshape(1, len(feature))
                featureVectors.append(feature)
                target = np.array(target).reshape(1, len(target))
                targetVectors.append(target)

        #Convert to numpy array
        featureVectors = np.array(featureVectors).reshape((len(featureVectors)), len(self.featureIntervalList))
        targetVectors = np.array(targetVectors).reshape((len(targetVectors)), len(self.targetIntervalList))

        return featureVectors, targetVectors

    def getProcessedData(self):
        return self.__generate__()


if __name__ == "__main__":
    df = pd.read_csv('testSeriesData.csv', index_col=0, parse_dates=True)
    series = pd.Series(data=df.as_matrix().flatten(),index=df.index)
    featureIntervalList = [pd.Timedelta(days=-3), pd.Timedelta(days=-2), pd.Timedelta(days=-1)]
    targetIntervalList = [pd.Timedelta(days=0)]
    tsp = TimeSeriesIntervalProcessor(series, featureIntervalList, targetIntervalList)
    featureVectors, targetVectors = tsp.getProcessedData()
    print(featureVectors,targetVectors)