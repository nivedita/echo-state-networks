from reservoir import classicESN as esn
import numpy as np
from utility import DataContinuousProcessor as processor
from reservoir import ReservoirTopology as topology
from performance import ErrorMetrics as errorMetrics

class HierarchicalESN:
    def __init__(self, featureTransformerParameters, predictorParameters,
                 inputData, outputData, depth):
        self.featureTransformerParameters = featureTransformerParameters
        self.predictorParameters = predictorParameters
        self.inputData = inputData
        self.outputData = outputData
        self.depth = depth

    def __createTransformer__(self, featureVectors):

        # Append the bias
        #featureVectors = np.hstack((np.ones((featureVectors.shape[0],1)),featureVectors))
        featureVectors = featureVectors
        targetVectors = self.outputData

        # Input weight matrix
        inputSize = featureVectors.shape[1]
        reservoirSize = self.featureTransformerParameters['size']
        inputWeightRandom = topology.RandomInputTopology(inputSize, reservoirSize, self.featureTransformerParameters['inputConnectivity']).generateWeightMatrix()
        reservoirWeightRandom = topology.RandomReservoirTopology(reservoirSize, self.featureTransformerParameters['reservoirConnectivity']).generateWeightMatrix()

        # Generate the reservoir
        self.transformer = esn.Reservoir(size=self.featureTransformerParameters['size'],
                            spectralRadius=self.featureTransformerParameters['spectralRadius'],
                            inputScaling=self.featureTransformerParameters['inputScaling'],
                            reservoirScaling=self.featureTransformerParameters['reservoirScaling'],
                            leakingRate=self.featureTransformerParameters['leakingRate'],
                            initialTransient=self.featureTransformerParameters['initialTransient'],
                            inputData=featureVectors,
                            outputData=targetVectors,
                            inputWeightRandom=inputWeightRandom,
                            reservoirWeightRandom=reservoirWeightRandom,
                            reservoirActivationFunction=self.featureTransformerParameters['reservoirActivation'],
                            outputActivationFunction=self.featureTransformerParameters['outputActivation'])

        # Collect and return the internal state
        internalStates = self.transformer.collectInternalStates(featureVectors)

        return internalStates

    def __collectInternalState_(self, reservoir, input):
        featureVector = input
        #featureVector = np.hstack((1.0,input[0, :])).reshape((1, input.shape[1]+1))

        # Collect the internal state from the reservoir
        return reservoir.collectInternalStates(featureVector)


    def trainReservoir(self):

        # Feature transformation
        features = self.__createTransformer__(self.inputData)

        # Append bias
        features = np.hstack((np.ones((features.shape[0],1)),features))

        # Generate the predictor
        # where features are transformed using transformer(esn)
        inputSize = features.shape[1]
        reservoirSize = self.predictorParameters['size']
        inputWeightRandom = topology.RandomInputTopology(inputSize, reservoirSize, self.predictorParameters['inputConnectivity']).generateWeightMatrix()
        reservoirWeightRandom = topology.RandomReservoirTopology(self.predictorParameters['size'], self.predictorParameters['reservoirConnectivity']).generateWeightMatrix()

        self.predictor = esn.Reservoir(size=self.predictorParameters['size'],
                                                spectralRadius=self.predictorParameters['spectralRadius'],
                                                inputScaling=self.predictorParameters['inputScaling'],
                                                reservoirScaling=self.predictorParameters['reservoirScaling'],
                                                leakingRate=self.predictorParameters['leakingRate'],
                                                initialTransient=self.predictorParameters['initialTransient'],
                                                inputData=features,
                                                outputData=self.outputData,
                                                inputWeightRandom=inputWeightRandom,
                                                reservoirWeightRandom=reservoirWeightRandom,
                                                reservoirActivationFunction=self.predictorParameters['reservoirActivation'],
                                                outputActivationFunction=self.predictorParameters['outputActivation'])
        # Train the predictor network
        self.predictor.trainReservoir()

    def predictOnePoint(self, input):

        # Here, transform the given input using transformer(esn)
        features = self.__collectInternalState_(self.transformer, input)

        # Append bias
        features = np.hstack((1.0,features[0, :])).reshape((1, features.shape[1]+1))

        # Predict
        return self.predictor.predict(features)





