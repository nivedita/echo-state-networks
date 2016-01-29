import numpy as np
from sklearn.metrics import mean_squared_error
#class RootMeanSquareError:
#	def compute(self, actual, predicted):
#   		N, D = actual.shape
#        error = 0
#        for d in range(D):
#        	error += (np.sum((actual[:N, d] - predicted[:N, d]) **2) / float(N)) ** 0.5
#        error = error / float(D)
#        return error

# class RootMeanSquareError:
#     def compute(self, actual, predicted):
#         rmse = mean_squared_error(actual, predicted)**0.5
#         return rmse

class RootMeanSquareError:
    def compute(self, actual, predicted):
        rmse = mean_squared_error(actual, predicted)
        return rmse

if __name__ == "__main__":
    error = RootMeanSquareError()
    error.compute(np.array([[1, 2],[3, 2]]), np.array([[4, 3],[3, 3]]))