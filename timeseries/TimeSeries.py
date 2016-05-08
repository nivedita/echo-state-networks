import numpy as np
class TimeSeriesProcessor:

    def __init__(self, inputData, depth, horizon):
        self.inputData = inputData
        self.horizon = horizon
        self.depth = depth

    def __generate__(self):
        # Create the dataset as numpy array
        rowCount = self.inputData.shape[0] - self.depth - self.horizon + 1
        colCount = self.depth + self.horizon #1 for current value
        processedData = np.zeros((rowCount, colCount))

        index = self.depth
        dataCount = 0
        while index <= (self.inputData.shape[0] - self.horizon):
            # Past values - including current
            depthValues = range(self.depth, 0, -1)
            colCount = 0
            for depth in depthValues:
                processedData[dataCount, colCount] = self.inputData[index - depth]
                colCount += 1

            # Future values
            horizonValues = range(0, self.horizon, 1)
            for horizon in horizonValues:
                processedData[dataCount, colCount] = self.inputData[index + horizon]
                colCount += 1

            index += 1
            dataCount += 1

        featureVectors = processedData[:,:self.depth]
        targetVectors = processedData[:,self.depth:]
        return featureVectors, targetVectors

    def getProcessedData(self):
        return self.__generate__()


if __name__ == "__main__":
    rawData = np.loadtxt("testData.txt", delimiter=',')
    tsp = TimeSeriesProcessor(rawData[:,4], 3, 1)
    processedData = tsp.getProcessedData()
    print(processedData)