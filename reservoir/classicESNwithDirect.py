import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.special import expit
from enum import Enum
from reservoir import ActivationFunctions as act

def _npRelu(np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

class Reservoir:
    def __init__(self, size, spectralRadius, inputScaling, reservoirScaling, leakingRate, initialTransient,
                 inputData, outputData, inputWeightRandom = None, reservoirWeightRandom = None,
                 reservoirActivationFunction=act.HyperbolicTangent(), outputActivationFunction=act.Linear()):
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

        # Generate the input and reservoir weights
        self.__generateInputWeight()
        self.__generateReservoirWeight()

        # Internal states
        self.internalState = np.zeros((self.inputN-self.initialTransient, self.Nx))
        self.latestInternalState = np.zeros(self.Nx)

        # Activation functions
        self.reservoirActivation = reservoirActivationFunction
        self.outputActivation = outputActivationFunction


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

    def collectInternalStates(self, inputData):

        inputN = inputData.shape[0]

        internalState = np.zeros(self.Nx)

        internalStates = np.zeros((inputN-self.initialTransient, self.Nx))

        # Compute internal states of the reservoir
        for t in range(inputN):
            term1 = np.dot(self.inputWeight,inputData[t])
            term2 = np.dot(self.reservoirWeight,internalState)
            internalState = (1.0-self.leakingRate)*internalState + self.leakingRate*self.reservoirActivation(term1 + term2)
            if t >= self.initialTransient:
                internalStates[t-self.initialTransient] = internalState
        return internalStates

    # TODO: This is a candidate for gnumpy conversion
    def trainReservoir(self):

        # Collect internal states
        self.internalState = self.collectInternalStates(self.inputData)

        # Learn the output weights
        #A = self.internalState
        A = np.hstack((self.inputData[self.initialTransient:], self.internalState))
        B = self.outputData[self.initialTransient:, :]

        # Solve for x in Ax = B
        for d in range(self.outputD):
            B = self.outputData[self.initialTransient:, d]
            self.outputWeight[d, :] = sla.lsmr(A, B, damp=1e-8)[0]

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
            internalState = (1-self.leakingRate)*internalState + self.leakingRate*self.reservoirActivation(term1 + term2)

            # Output
            output = self.outputActivation(np.dot(self.outputWeight, np.hstack((testInputData[t], internalState))))
            #output = np.dot(self.outputWeight, np.hstack((testInputData[t], internalState)))
            testOutputData[t, :] = output

        # This is to preserve the internal state between multiple predict calls
        self.latestInternalState = internalState

        return testOutputData

    # TODO: This is a candidate for gnumpy conversion
    def predictOnePoint(self, testInput):
        term1 = np.dot(self.inputWeight,testInput[0])
        term2 = np.dot(self.reservoirWeight,self.latestInternalState)
        self.latestInternalState = (1-self.leakingRate)*self.latestInternalState + self.leakingRate*self.reservoirActivation(term1 + term2)

        # Output - Non-linearity applied through activation function
        #output = self.outputActivation(np.dot(self.outputWeight, self.latestInternalState))
        output = self.outputActivation(np.dot(self.outputWeight, np.hstack((testInput[0], self.latestInternalState))))
        return output