from reservoir import classicESN as esn
import numpy as np
from reservoir import ReservoirTopology as topology
from performance import ErrorMetrics as errorMetrics

class HierarchicalESN:
    def __init__(self, lowerLayerParameters, higherLayerParameters,
                 featureIndicesList, inputData, outputData):
        self.lowerLayerParameters = lowerLayerParameters
        self.higherLayerParameters = higherLayerParameters
        self.featureIndicesList = featureIndicesList
        self.inputData = inputData
        self.outputData = outputData

        # List to store the reservoirs
        self.lowerLayerNetworks = []

        # Other member variables
        self.lowerLayerCount = len(self.featureIndicesList)

    def __collectOutputs__(self, featureIndices):

        # Form the feature and target vectors
        featureVectors = self.__formFeaturevectors__(self.inputData, featureIndices)
        targetVectors = self.outputData

        # Append bias
        featureVectors = np.hstack((np.ones((featureVectors.shape[0], 1)),featureVectors))

        # Input weight matrix
        inputSize = featureVectors.shape[1]
        reservoirSize = self.lowerLayerParameters['size']
        inputWeightRandom = topology.RandomInputTopology(inputSize, reservoirSize, self.lowerLayerParameters['inputConnectivity']).generateWeightMatrix()
        reservoirWeightRandom = topology.RandomReservoirTopology(reservoirSize, self.lowerLayerParameters['reservoirConnectivity']).generateWeightMatrix()

        # Generate the reservoir
        res = esn.Reservoir(size=self.lowerLayerParameters['size'],
                            spectralRadius=self.lowerLayerParameters['spectralRadius'],
                            inputScaling=self.lowerLayerParameters['inputScaling'],
                            reservoirScaling=self.lowerLayerParameters['reservoirScaling'],
                            leakingRate=self.lowerLayerParameters['leakingRate'],
                            initialTransient=self.lowerLayerParameters['initialTransient'],
                            inputData=featureVectors,
                            outputData=targetVectors,
                            inputWeightRandom=inputWeightRandom,
                            reservoirWeightRandom=reservoirWeightRandom,
                            reservoirActivationFunction=self.lowerLayerParameters['reservoirActivation'],
                            outputActivationFunction=self.lowerLayerParameters['outputActivation'])

        # Train the reservoir
        res.trainReservoir()

        # Just assign the weights randomly

        # Store the reservoir
        self.lowerLayerNetworks.append(res)

        # Collect the outputs
        outputs = res.predict(featureVectors)

        return outputs

    def __collectOutput_(self, reservoir, input, featureIndices):
        # Form the feature
        featureVector = input[:, featureIndices]

        # Append bias
        featureVector = np.hstack((1.0,featureVector[0, :])).reshape((1, featureVector.shape[1]+1))

        # Collect the output
        return reservoir.predictOnePoint(featureVector)

    def __formFeaturevectors__(self, inputData, featureIndices):
        featureVectors = inputData[:, featureIndices]
        return featureVectors

    def trainReservoir(self):

        # Features for the network in the higher layer
        features = None

        # Collect outputs from the lower layer
        for i in range(self.lowerLayerCount):
            if(features is None): # First time
                features = self.__collectOutputs__(self.featureIndicesList[i])
            else:
                features = np.hstack((features, self.__collectOutputs__(self.featureIndicesList[i])))

        # Append bias
        features = np.hstack((np.ones((features.shape[0],1)),features))

        # Generate the higher layer reservoir
        # where features are the outputs of the lower layer networks
        inputSize = features.shape[1]
        reservoirSize = self.higherLayerParameters['size']

        inputWeightRandom = topology.RandomInputTopology(inputSize, reservoirSize, self.higherLayerParameters['inputConnectivity']).generateWeightMatrix()
        reservoirWeightRandom = topology.RandomReservoirTopology(self.higherLayerParameters['size'], self.higherLayerParameters['reservoirConnectivity']).generateWeightMatrix()

        self.higherLayerReservoir = esn.Reservoir(size=self.higherLayerParameters['size'],
                                                spectralRadius=self.higherLayerParameters['spectralRadius'],
                                                inputScaling=self.higherLayerParameters['inputScaling'],
                                                reservoirScaling=self.higherLayerParameters['reservoirScaling'],
                                                leakingRate=self.higherLayerParameters['leakingRate'],
                                                initialTransient=self.higherLayerParameters['initialTransient'],
                                                inputData=features,
                                                outputData=self.outputData,
                                                inputWeightRandom=inputWeightRandom,
                                                reservoirWeightRandom=reservoirWeightRandom,
                                                reservoirActivationFunction=self.higherLayerParameters['reservoirActivation'],
                                                outputActivationFunction=self.higherLayerParameters['outputActivation'])
        # Train the composite network
        self.higherLayerReservoir.trainReservoir()

    def predictOnePoint(self, input):

        # Here, you have to give the input to all the lower layer reservoirs and
        # collect the outputs as a feature vector

        # Features for the composite network
        features = None

        # Collect all internal states
        for i in range(self.lowerLayerCount):
            if(features is None): # First time
                features = self.__collectOutput_(self.lowerLayerNetworks[i], input, self.featureIndicesList[i])
            else:
                features = np.hstack((features, self.__collectOutput_(self.lowerLayerNetworks[i], input, self.featureIndicesList[i])))

        # Append bias
        features = np.hstack((1.0,features)).reshape(1,self.lowerLayerCount+1)

        # Predict
        return self.higherLayerReservoir.predictOnePoint(features)







