
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from reservoir import ReservoirTopology as topology

class EchoStateNetwork:
    def __init__(self, size, inputData, outputData, reservoirTopology, spectralRadius=0.80, inputScaling=0.5, reservoirScaling=0.5, leakingRate=0.3,
                 initialTransient=0, inputConnectivity=0.6, inputWeightConn=None, reservoirWeightConn=None):
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
        self.reservoirTopology = reservoirTopology
        self.inputConnectivity = inputConnectivity

        #Initialize weights
        self.inputN, self.inputD = self.inputData.shape
        self.outputN, self.outputD = self.outputData.shape
        self.Nu = self.inputD
        self.Ny = self.outputD
        self.inputWeight = np.zeros((self.Nx, self.Nu))
        self.reservoirWeight = np.zeros((self.Nx, self.Nx))
        self.outputWeight = np.zeros((self.Ny, self.Nx))

        #Generate the input and reservoir weights
        if(inputWeightConn == None):
            self.__generateInputWeight()
        else:
            self.inputWeightRandom, self.randomInputIndices = inputWeightConn
            self.inputWeight = self.inputWeightRandom
            self.__applyInputScaling()

        if(reservoirWeightConn == None):
            self.__generateReservoirWeight()
        else:
            self.reservoirWeightRandom, self.randomReservoirIndices = reservoirWeightConn
            self.reservoirWeight = self.reservoirWeightRandom
            self.__applyReservoirScaling()

        #Internal states
        self.internalState = np.zeros((self.inputN-self.initialTransient, self.Nx))
        self.latestInternalState = np.zeros(self.Nx)

    def __generateInputConnectivityMatrix(self):
        #Initialize the matrix to zeros
        connectivity = np.zeros((self.Nx, self.Nu))

        indices1 = []
        indices2 = []
        for i in range(self.Nx):
            indices = np.random.choice(self.Nu, size=int(self.inputConnectivity * self.Nu), replace=False)
            connectivity[i, indices] = 1.0
            for j in range(indices.shape[0]):
                indices1.append(i)
                indices2.append(indices[j])
        randomIndices = np.array(indices1), np.array(indices2)
        return connectivity, randomIndices

    def __generateInputWeight(self):
        #Choose a uniform distribution and adjust it according to the input scaling
        #ie. the values are chosen from [-inputScaling, +inputScaling]
        #TODO: Normalize ?
        self.inputWeightRandom = np.random.rand(self.Nx, self.Nu)
        self.inputConnMatrix, self.randomInputIndices = self.__generateInputConnectivityMatrix()
        self.inputWeightRandom = self.inputWeightRandom * self.inputConnMatrix
        self.inputWeight = self.inputWeightRandom

        #Scaling
        self.__applyInputScaling()

    def __applyInputScaling(self):
        # Scaling
        inputScaling = np.zeros((self.Nx, self.Nu))
        inputScaling[self.randomInputIndices] = self.inputScaling
        self.inputWeight = self.inputWeight - inputScaling

    def __generateReservoirWeight(self):
        #Choose a uniform distribution
        #TODO: Normalize ?
        self.reservoirWeightRandom = np.random.rand(self.Nx, self.Nx)
        self.reservoirConnMatrix, self.randomReservoirIndices = self.reservoirTopology.generateConnectivityMatrix()
        self.reservoirWeightRandom = self.reservoirWeightRandom * self.reservoirConnMatrix
        self.reservoirWeight = self.reservoirWeightRandom

        # Scaling
        self.__applyReservoirScaling()

    def __applyReservoirScaling(self):
        # Scaling
        reservoirScaling = np.zeros((self.Nx, self.Nx))
        reservoirScaling[self.randomReservoirIndices] = self.reservoirScaling
        self.reservoirWeight = self.reservoirWeight - reservoirScaling

        #Make the reservoir weight matrix - a unit spectral radius
        rad = np.max(np.abs(la.eigvals(self.reservoirWeight)))
        self.reservoirWeight = self.reservoirWeight / rad

        #Force spectral radius
        self.reservoirWeight = self.reservoirWeight * self.spectralRadius

    def trainReservoir(self):

        internalState = np.zeros(self.Nx)

        #Compute internal states of the reservoir
        for t in range(self.inputN):
            term1 = np.dot(self.inputWeight,self.inputData[t])
            term2 = np.dot(self.reservoirWeight,internalState)
            internalState = (1.0-self.leakingRate)*internalState + self.leakingRate*np.tanh(term1 + term2)
            if t >= self.initialTransient:
                self.internalState[t-self.initialTransient] = internalState

        #Learn the output weights
        A = self.internalState
        B = self.outputData[self.initialTransient:, :]

        #Solve for x in Ax = B
        for d in range(self.outputD):
            B = self.outputData[self.initialTransient:, d]
            self.outputWeight[d, :] = sla.lsmr(A, B, damp=1e-8)[0]

    def predict(self, testInputData):

        testInputN, testInputD = testInputData.shape
        statesN, resD = self.internalState.shape

        internalState = self.latestInternalState

        testOutputData = np.zeros((testInputN, self.outputD))

        for t in range(testInputN):
            #reservoir activation
            term1 = np.dot(self.inputWeight,testInputData[t])
            term2 = np.dot(self.reservoirWeight,internalState)
            internalState = (1-self.leakingRate)*internalState + self.leakingRate*np.tanh(term1 + term2)

            #compute output
            output = np.dot(self.outputWeight, internalState)
            testOutputData[t, :] = output

        #This is to preserve the internal state between multiple predict calls
        self.latestInternalState = internalState

        return testOutputData



