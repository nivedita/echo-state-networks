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
                 inputData, outputData, batchLearnRatio = 0.95, regFactor = 0.001, forgettingParameter = 0.999, inputWeightRandom = None, reservoirWeightRandom = None,
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
        self.batchLearnRatio = batchLearnRatio
        self.regFactor = regFactor
        self.forgettingParameter = forgettingParameter

        # Initialize weights
        self.inputN, self.inputD = self.inputData.shape
        self.outputN, self.outputD = self.outputData.shape
        self.Nu = self.inputD
        self.Ny = self.outputD
        self.inputWeight = np.zeros((self.Nx, self.Nu))
        self.reservoirWeight = np.zeros((self.Nx, self.Nx))
        self.outputWeight = np.zeros((self.Nx, self.Ny))

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

        # Online Training related items
        a = np.random.random_integers(900000, 999999) # Large value
        self.errorCovariance = a * np.identity(size)


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


    def trainReservoir(self):

        # Divide the training data into batch and online
        nBatch = int(self.batchLearnRatio*self.inputN)
        inputDataBatch = self.inputData[:nBatch]
        outputDataBatch = self.outputData[:nBatch]
        inputDataOnline = self.inputData[nBatch:]
        outputDataOnline = self.outputData[nBatch:]

        # Train batch and update the error covariance matrix
        self.trainReservoirBatch(inputDataBatch, outputDataBatch)

        # Train the rest of the points using incremental update
        self.trainReservoirOnline(inputDataOnline, outputDataOnline)

        # Reset the internal state
        self.latestInternalState = np.zeros(self.Nx)

    def trainReservoirBatch(self, inputData, outputData):

        # Collect the reservoir states
        inputN = inputData.shape[0]
        internalStates = np.zeros((inputN-self.initialTransient, self.Nx))
        internalState = np.zeros(self.Nx)
        for t in range(inputN):
            term1 = np.dot(self.inputWeight, inputData[t])
            term2 = np.dot(self.reservoirWeight,internalState)
            internalState = (1.0-self.leakingRate)*internalState + self.leakingRate*self.activation(term1 + term2)
            if t >= self.initialTransient:
                internalStates[t-self.initialTransient] = internalState
                self.latestInternalState = internalState

        # # Solve for x in Ax = B
        # A = internalStates
        # for d in range(self.outputD):
        #     B = outputData[self.initialTransient:, d]
        #     self.outputWeight[:, d] = sla.lsmr(A, B, damp=1e-8)[0]

        # Compute the error covariance matrix
        self.errorCovariance = np.linalg.inv(np.dot(internalStates.T, internalStates) + self.regFactor * np.identity(self.Nx))

        # Compute the output weights
        self.outputWeight = np.dot(self.errorCovariance, np.dot(internalStates.T, outputData[self.initialTransient:]))

    # TODO: This is a candidate for gnumpy conversion
    def trainReservoirOnline(self, inputData, outputData):

        internalState = self.latestInternalState

        inputN = inputData.shape[0]

        # Compute internal states of the reservoir
        for t in range(inputN):
            print("Processing.."+str(t))
            term1 = np.dot(self.inputWeight,inputData[t])
            term2 = np.dot(self.reservoirWeight,internalState)
            internalState = (1.0-self.leakingRate)*internalState + self.leakingRate*self.activation(term1 + term2)
            if t >= self.initialTransient:
                # Output
                x = np.array(internalState).reshape((self.Nx),1)
                output = np.dot(self.outputWeight.T, x)

                # Error
                error = outputData[t] - output

                # Innovation vector
                innovationvector = np.dot(self.errorCovariance, x) / (np.dot(x.T, np.dot(self.errorCovariance, x)) + self.forgettingParameter)

                # Update the weights
                self.outputWeight += np.dot(innovationvector, error)

                # Update the error covariance
                self.errorCovariance = (self.errorCovariance - np.dot(np.dot(innovationvector, x.T), self.errorCovariance)) / self.forgettingParameter


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
            output = np.dot(self.outputWeight.T, internalState)
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
        output = np.dot(self.outputWeight.T, self.latestInternalState)
        # Apply Relu to output
        if(self.relu):
            output = _npRelu(output)
        return output