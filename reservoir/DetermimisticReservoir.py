
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

class DeterministicReservoir:
    def __init__(self, size, inputWeight_r, reservoirWeight_w, inputData, outputData, leakingRate, initialTransient):
        """

        :param size: size of the reservoir
        :param inputWeight_r: r of the input weight
        :param reservoirWeight_r: w of the reservoir weight
        :param inputData: input data (N X D)
        :param outputData: output data (N X D)
        :param leakingRate: leaking rate of the reservoir
        :param initialTransient: burin/warmp/washout time
        """
        self.Nx = size
        self.inputWeight_r = inputWeight_r
        self.reservoirWeight_r = reservoirWeight_w
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

        #Generate the input and reservoir weights
        self.__generateInputWeight()
        self.__generateReservoirWeight()

        #Internal states
        self.internalState = np.zeros((self.inputN-self.initialTransient, self.Nx))
        self.latestInternalState = None


    def __generateInputWeight(self):
        #TODO: How to generate the weights for this mimimum complexity ESN

    def __generateReservoirWeight(self):
        #TODO: How to generate the weights for this minimum complexity ESN

    #I hope the training and predicting reservoirs are same as in standard ESN
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



