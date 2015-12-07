import numpy as np
from scipy import linalg


class DataWhittener():
	
	def __init__(self, ZCA = True):
		"""Whitten Data in order to de-correlate and normalize the data 
		Set ZCA = False if you don't want to rotate the data back to original feature space"""
		
		self.ZCA = ZCA												

	def WhittenData(self, data):
		"""This function expects data to be in (N X D) Numpy Array"""
		
		self.meanDataPoint = np.mean(data,axis=0)
		X = data - self.meanDataPoint      							#Convert data to have Zero Mean
		X = X.T                                   
		Xcov=np.dot(X, X.T)											#Compute Covariance matrix		
		val, self.vec = np.linalg.eigh(Xcov)						#Do Eigen Vector Decomposition for Doing PCA/ZCA Whittening
		self.L = np.linalg.inv(linalg.sqrtm(np.diag(val+0.000001)))	#Scale the Eigen Values to have unit Variance        	
		X = np.dot(self.vec.T, X)                					#Rotates towards the Priniciple Axis
		X = np.dot(self.L, X)										#Scales the data to have unit variance(PCA Whittening)
		if(self.ZCA == True):
			X = np.dot(self.vec, X) 								#Rotates back to original Space(ZCA Whittening)
	
		return np.real(X.T) 										#Enforcing the data not to be complex numbers
	
	def WhittenNewDataPoints(self, data):
		"""With the computed Basis Vectors, Whitten new data points"""
		
		X = data - self.meanDataPoint								#Convert data to have Zero Mean
		X = X.T
		X = np.dot(self.vec.T , X)									#Rotates towards the Priniciple Axis
		X = np.dot(self.L, X)										#Scales the data to have unit variance(PCA Whittening)
		if(self.ZCA == True):
			X = np.dot(self.vec, X)                                 #Rotates back to original Space(ZCA Whittening)
		
		return np.real(X.T) 										#Enforcing the data not to be complex numbers
	
	def getBasisVectorsForWhittening(self):
		"""Get Basis Vectors for whittening data in future or exporting them"""
		
		return self.meanDataPoint, self.vec, self.L
	
	def UnwhittenData(self, data):
		"""Unwhittening to reverse the normalization done"""

		X = data.T
		if(self.ZCA == True):
			X = np.dot(np.linalg.pinv(self.vec), X)
		X = np.dot(np.linalg.pinv(self.L), X)
		X = np.dot(np.linalg.pinv(self.vec.T), X)
		X = X.T
		X = X + self.meanDataPoint

		return X
		
		
		
