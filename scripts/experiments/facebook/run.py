#Run this script to run the experiment
#Steps to follow are:
# 1. Preprocessing of data
# 2. Give the data to the reservoir
# 3. Plot the performance (such as error rate/accuracy)

from reservoir import Reservoir as reservoir
from plotting import OutputPlot as outputPlot, ErrorPlot as errorPlot
from performance import RootMeanSquareError as rmse
import numpy as np
import json

#Read data from the file
with open('fans_by_country.json') as data_file:
    data = json.load(data_file)

#Taylor Swift
data = data['11307']


print("Done!")