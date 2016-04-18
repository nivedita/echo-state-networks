import numpy as np
class DataContinuosProcessor:

    def __init__(self, data, depth, horizon):
        self.data = data
        self.depth = depth
        self.horizon = horizon

    def __generate__(self):

        data = self.data
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
    data = np.loadtxt('testSeriesData.txt')
    tsp = DataContinuosProcessor(data, depth=3, horizon=1)
    featureVectors, targetVectors = tsp.getProcessedData()
    print(featureVectors,targetVectors)