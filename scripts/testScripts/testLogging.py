import pickle
import numpy as np

data1 = [(np.array([1,2]), 0.5, {"degree":9, "diameter":10}), (np.array([1,4]), 0.25, {"degree":9, "diameter":10}), (np.array([4,2]), 0.15, {"degree":9, "diameter":10})]

output = open('data.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(data1, output)

output.close()

pkl_file = open('data.pkl', 'rb')

data2 = pickle.load(pkl_file)

print("Data is:"+str(data2))

pkl_file.close()