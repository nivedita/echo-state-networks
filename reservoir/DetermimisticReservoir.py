
import numpy as np
import scipy.sparse.linalg as sla
from sympy.mpmath import mp

#Delay Line Reservoir
class ReservoirTopologyDLR:
    def __init__(self, reservoirWeight):
        self.r = reservoirWeight

    def generateWeightMatrix(self, Nx):
        reservoirWeightMatrix = np.zeros((Nx, Nx))
        for i in range(0, Nx-1):
            reservoirWeightMatrix[i+1, i] = self.r
        return reservoirWeightMatrix

#Delay Line Reservoir with feedback
class ReservoirTopologyDLRB:
    def __init__(self, reservoirForwardWeight, reservoirFeedbackWeight):
        self.r = reservoirForwardWeight
        self.b = reservoirFeedbackWeight

    def generateWeightMatrix(self, Nx):
        reservoirWeightMatrix = np.zeros((Nx, Nx))
        for i in range(0, Nx-1):
            reservoirWeightMatrix[i+1, i] = self.r
            reservoirWeightMatrix[i, i+1] = self.b
        return reservoirWeightMatrix

#Simple Cycle Reservoir
class ReservoirTopologySCR:
    def __init__(self, reservoirWeight):
        self.r = reservoirWeight

    def generateWeightMatrix(self, Nx):
        reservoirWeightMatrix = np.zeros((Nx, Nx))
        for i in range(0, Nx-1):
            reservoirWeightMatrix[i+1, i] = self.r
        reservoirWeightMatrix[0, Nx-1] = self.r
        return reservoirWeightMatrix

class DeterministicReservoir:
    def __init__(self, size, inputWeight_v, inputWeightScaling, inputData, outputData, leakingRate, initialTransient, reservoirTopology):
        """

        :param size: size of the reservoir
        :param inputWeight_v: v of the input weight
        :param inputData: input training data (N X D)
        :param outputData: output training data (N X D)
        :param leakingRate: leaking rate of the reservoir
        :param initialTransient: warmup time
        """
        self.Nx = size
        self.inputWeight_v = inputWeight_v
        self.inputWeightScaling = inputWeightScaling
        self.leakingRate = leakingRate
        self.initialTransient = initialTransient
        self.inputData = inputData
        self.outputData = outputData

        #Initialize weights
        self.inputN, self.inputD = self.inputData.shape
        self.outputN, self.outputD = self.outputData.shape
        self.Nu = self.inputD
        self.Ny = self.outputD
        self.inputWeight = np.zeros((self.Nx, self.Nu))
        self.reservoirWeight = np.zeros((self.Nx, self.Nx))
        self.outputWeight = np.zeros((self.Ny, self.Nx))

        #Constant - pi for sign generation
        mp.dps = self.Nx+2 # number of digits
        self.pi = str(mp.e)[2:]

        #Reservoir Topology
        self.reservoirTopology = reservoirTopology

        #Generate the input and reservoir weights
        self.__generateInputWeight()
        self.__generateReservoirWeight()

        #Internal states
        self.internalState = np.zeros((self.inputN-self.initialTransient, self.Nx))
        self.latestInternalState = None

    def __generateInputWeight(self):
        #TODO: How to generate the weights for this mimimum complexity ESN
        for i in range(self.Nu):
            for d in range(self.Nx):
                if 0 <= int(self.pi[d]) <= 4:
                    self.inputWeight[d, i] = -1.0 * self.inputWeight_v
                if 5 <= int(self.pi[d]) <= 9:
                    self.inputWeight[d, i] = 1.0 * self.inputWeight_v

        #Input scaling
        self.inputWeight = self.inputWeight - self.inputWeightScaling

    def __generateReservoirWeight(self):
        self.reservoirWeight = self.reservoirTopology.generateWeightMatrix(self.Nx)

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
        if(self.latestInternalState == None):
            internalState = np.zeros(self.Nx)
        else:
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

        self.latestInternalState = internalState

        return testOutputData



