from sklearn.svm import OneClassSVM as detector
import numpy as np

det = detector()

X = np.array([1,2,3,4,5,6,7,8,9,10,1000]).reshape(11,1)

det.fit(X)

Y = det.predict(100)

print(Y)






