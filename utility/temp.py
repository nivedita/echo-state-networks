from scipy.stats import pearsonr, spearmanr
def getCorrelationMatrix(self, featureVectors, targetVectors, reservoirSize):
    inputSize = featureVectors.shape[1]
    correlationCoefficient = self.getCorrelationCoefficients(featureVectors, targetVectors)

    correlationMatrix = [np.random.rand(1,reservoirSize)]
    for i in range(inputSize):
        correlation = correlationCoefficient[0,i]
        correlationMatrix.append(np.ones((1, reservoirSize)) * correlation)
    correlationMatrix = np.array(correlationMatrix).reshape((inputSize+1, reservoirSize)).T
    return correlationMatrix

def getCorrelationCoefficients(self, featureVectors, targetVectors):
    correlations = []
    y = targetVectors[:, 0]
    # For each feature vector, calculate the correlation coefficient
    for i in range(featureVectors.shape[1]):
        x = featureVectors[:, i]
        #correlation, p_value = pearsonr(x,y)
        correlation, p_value = spearmanr(x,y)
        correlations.append(correlation)
    # Scale the correlations
    correlations = np.array(correlations)

    # Re-shape
    correlations = correlations.reshape((1, featureVectors.shape[1]))
    return correlations