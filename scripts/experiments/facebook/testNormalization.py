from sklearn import preprocessing as pp
import numpy as np

maxAbs = pp.MinMaxScaler((-1,+1))
rawData = np.loadtxt("facebookFansHistory_bmw_raw.txt", delimiter=',')
dataForTraining = rawData[:, 4]
transformedMaxAbs = maxAbs.fit_transform(dataForTraining)

print(maxAbs.inverse_transform([0.668035875102]))

