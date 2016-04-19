from sklearn.svm import OneClassSVM as detector
import numpy as np
from utility import Utility
from sklearn.covariance import EllipticEnvelope

#det = detector()
det = EllipticEnvelope()

# X = np.array([1,2,3,4,5,6,7,8,9,10,1000]).reshape(11,1)
#
# det.fit(X)
#
# Y = det.predict(100)
#
# print(Y)

# Find outliers in the interaction rate data

# Step 1 - Convert the dataset into pandas series
util = Utility.SeriesUtility()
datasetFileName = "fans_change_taylor_swift.csv"
series = util.convertDatasetsToSeries(datasetFileName)

series = util.resampleSeriesSum(series, "D")

numberOfPoints = series.data.shape[0]
X = series.values.flatten().reshape(numberOfPoints,1)

det.fit(X)

predicted = det.predict(X)

for i in range(numberOfPoints):
    outputClass = det.predict(X[i])[0]

    if(outputClass == -1):
        print("Outlier detected...")







