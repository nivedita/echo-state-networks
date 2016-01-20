import numpy as np
import pandas as pd
class TimeSeriesContinuosProcessor:

    def __init__(self, series, depth, horizon):
        self.series = series
        self.depth = depth
        self.horizon = horizon

    def __generate__(self):

        data = self.series.values
        trainingVectors = []

        size = self.depth + self.horizon
        for i in range(size):
            # Lose the last few elements (for re-shaping)
            loseCount =  data.shape[0] % (size)
            b = data[:data.shape[0]-loseCount]

            # Reshape and form feature vectors
            b = b.reshape((data.shape[0] / size, size)).tolist()
            trainingVectors.extend(b)

            # Move the array (like sliding window)
            data = data[1:]

        # Separate the feature and target vectors
        trainingArray = np.array(trainingVectors)
        featureVectors = trainingArray[:,:self.depth]
        targetVectors = trainingArray[:,self.depth:]

        return featureVectors, targetVectors

    def getProcessedData(self):
        return self.__generate__()


if __name__ == "__main__":
    df = pd.read_csv('testSeriesData.csv', index_col=0, parse_dates=True)
    series = pd.Series(data=df.as_matrix().flatten(),index=df.index)
    tsp = TimeSeriesContinuosProcessor(series, depth=3, horizon=1)
    featureVectors, targetVectors = tsp.getProcessedData()
    print(featureVectors,targetVectors)