import numpy as np
import pandas as pd
class TimeSeriesDepthIntervalProcessor:

    def __init__(self, data, depth, period):
        self.data = data
        self.depth = depth
        self.period = period

    def __generate__(self):
        data = self.data

        d = self.depth
        p = self.period

        feature = np.empty((0,p+1), float)

        totalSize = data.shape[0]
        loseSize = data.shape[0] % d
        temp = data[:data.shape[0] - loseSize]

        temp = temp.reshape((d,int(data.shape[0]/d))).T
        print(temp.shape)
        feature = np.vstack((feature, temp))

        print(feature)

    def getProcessedData(self):
        return self.__generate__()


if __name__ == "__main__":
    # df = pd.read_csv('testSeriesData.csv', index_col=0, parse_dates=True)
    # series = pd.Series(data=df.as_matrix().flatten(),index=df.index)
    tsp = TimeSeriesDepthIntervalProcessor(np.arange(100), depth=7, period=3)
    featureVectors, targetVectors = tsp.getProcessedData()
    print(featureVectors,targetVectors)