import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.special import expit
from enum import Enum

class ActivationFunction(Enum):
    TANH = 1
    EXPIT = 2
    ReLU = 3


def _npRelu(np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

class Reservoir:
    def __init__(self, size, spectralRadius, inputScaling, reservoirScaling, leakingRate, initialTransient,
                 inputData, outputData, inputWeightRandom = None, reservoirWeightRandom = None,
                 activationFunction=ActivationFunction.TANH, outputRelu = False):
        """
        :param Nx: size of the reservoir
        :param spectralRadius: spectral radius for reservoir weight matrix
        :param inputScaling: scaling for input weight matrix - the values are chosen from [-inputScaling, +inputScaling]
                                1 X D - For each parameter
        :param leakingRate: leaking rate of the reservoir
        :param data: input data (N X D)
        """
        self.Nx = size
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.reservoirScaling = reservoirScaling
        self.leakingRate = leakingRate
        self.initialTransient = initialTransient
        self.inputData = inputData
        self.outputData = outputData

        # Initialize weights
        self.inputN, self.inputD = self.inputData.shape
        self.outputN, self.outputD = self.outputData.shape
        self.Nu = self.inputD
        self.Ny = self.outputD
        self.inputWeight = np.zeros((self.Nx, self.Nu))
        self.reservoirWeight = np.zeros((self.Nx, self.Nx))
        self.outputWeight = np.zeros((self.Ny, (self.Nu + self.Nx)))

        if(inputWeightRandom is None):
            self.inputWeightRandom = np.random.rand(self.Nx, self.Nu)
        else:
            self.inputWeightRandom = np.copy(inputWeightRandom)
        if(reservoirWeightRandom is None):
            self.reservoirWeightRandom = np.random.rand(self.Nx, self.Nx)
        else:
            self.reservoirWeightRandom = np.copy(reservoirWeightRandom)

        # Output Relu
        self.relu = outputRelu

        # Generate the input and reservoir weights
        self.__generateInputWeight()
        self.__generateReservoirWeight()

        # Internal states
        self.internalState = np.zeros((self.inputN-self.initialTransient, self.Nx))
        self.latestInternalState = np.zeros(self.Nx)

        # Activation Function
        if activationFunction == ActivationFunction.TANH:
            self.activation = np.tanh
        elif activationFunction == ActivationFunction.EXPIT:
            self.activation = expit
        elif activationFunction == ActivationFunction.ReLU:
            self.activation = _npRelu


    def __generateInputWeight(self):
        # Choose a uniform distribution and adjust it according to the input scaling
        # ie. the values are chosen from [-inputScaling, +inputScaling]
        self.inputWeight = self.inputWeightRandom

        # Apply scaling only non-zero elements (Because of various toplogies)
        self.inputWeight[self.inputWeight!=0.0] = self.inputWeight[self.inputWeight!=0.0] - self.inputScaling

    def __generateReservoirWeight(self):
        # Choose a uniform distribution
        self.reservoirWeight = self.reservoirWeightRandom

        # Apply scaling only non-zero elements (Because of various toplogies)
        self.reservoirWeight[self.reservoirWeight!=0.0] = self.reservoirWeight[self.reservoirWeight!=0.0] - self.reservoirScaling

        # Make the reservoir weight matrix - a unit spectral radius
        rad = np.max(np.abs(la.eigvals(self.reservoirWeight)))
        self.reservoirWeight = self.reservoirWeight / rad

        # Force spectral radius
        self.reservoirWeight = self.reservoirWeight * self.spectralRadius

    # TODO: This is a candidate for gnumpy conversion
    # This is where the output weights are adapted in an online fashion
    def trainReservoir(self):

        x_n = np.zeros((1, self.Nx))

        adaptationRate = 0.0001


        # Compute internal states of the reservoir
        for t in range(self.inputN):
            # Input vector u[n]
            u_n = self.inputData[t].reshape((1, self.Nu))

            # Reservoir internal state
            term1 = np.dot(u_n, self.inputWeight.T)
            term2 = np.dot(x_n, self.reservoirWeight)
            x_n = (1.0-self.leakingRate)*x_n + self.leakingRate*self.activation(term1 + term2)

             # Expected output
            d_n = self.outputData[t]

            # Calculate the output of the network
            con = np.hstack((u_n, x_n))
            y_n = np.dot(con, self.outputWeight.T)

            # Error
            e_n = d_n - y_n
            print(e_n)

            # Adapt the output weights
            correctionTerm = adaptationRate * np.dot(e_n.T, con)
            self.outputWeight = self.outputWeight + correctionTerm


    # TODO: This is a candidate for gnumpy conversion
    def predict(self, testInputData):

        testInputN, testInputD = testInputData.shape
        statesN, resD = self.internalState.shape

        internalState = self.latestInternalState

        testOutputData = np.zeros((testInputN, self.outputD))

        for t in range(testInputN):
            # Reservoir activation
            term1 = np.dot(self.inputWeight,testInputData[t])
            term2 = np.dot(self.reservoirWeight,internalState)
            internalState = (1-self.leakingRate)*internalState + self.leakingRate*self.activation(term1 + term2)

            # Output
            output = np.dot(self.outputWeight, np.hstack((testInputData[t], internalState))) # TODO: update here
            # Apply Relu to output
            if(self.relu):
                output = _npRelu(output)
            testOutputData[t, :] = output

        # This is to preserve the internal state between multiple predict calls
        self.latestInternalState = internalState

        return testOutputData

    # TODO: This is a candidate for gnumpy conversion
    def predictOnePoint(self, testInput):
        term1 = np.dot(self.inputWeight,testInput[0])
        term2 = np.dot(self.reservoirWeight,self.latestInternalState)
        self.latestInternalState = (1-self.leakingRate)*self.latestInternalState + self.leakingRate*self.activation(term1 + term2)

        # Output - Non-linearity applied through activation function
        output = np.dot(self.outputWeight, np.hstack((testInput[0], self.latestInternalState)))
        # Apply Relu to output
        if(self.relu):
            output = _npRelu(output)
        return output