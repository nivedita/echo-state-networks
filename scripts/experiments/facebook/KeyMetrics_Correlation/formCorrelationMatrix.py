from utility import Utility
from datetime import datetime
from plotting import CorrelationMatrix as mplot
from scipy.stats import pearsonr, spearmanr
import os
import decimal



# Step 1 - Read all the datasets into an array
util = Utility.SeriesUtility()
metricNames = ["Average Interactions Per Post", "Average PTAT Rate", "Fans Change Rate", "Fans Total", "Interaction Rate", "Own Posts"]
dataSetFileNames = ["Average_Interactions_Per_Post.csv", "Average_PTAT_Rate.csv", "Fans_Change_Rate.csv", "Fans_Total.csv", "Interaction_Rate.csv", "Own_Posts.csv"]
metricData = []

for name in dataSetFileNames:
    data = util.convertDatasetsToSeries(name).values.flatten()
    metricData.append(data)


# Step 2: Calculate the correlation for each combination of metrices
correlationData = "["
for i in range(len(metricData)):
    metric1 = metricData[i]
    for j in range(len(metricData)):
        metric2 = metricData[j]
        #correlation, p_value = pearsonr(metric1, metric2)
        correlation, p_value = spearmanr(metric1, metric2)
        correlation = round(decimal.Decimal(correlation),3)
        correlationData = correlationData + "["+str(i)+","+str(j)+","+str(correlation)+"],"
correlationData = correlationData[0:len(correlationData) - 1]
correlationData = correlationData + "]"

# Step 3: Plot the correlation matrix
folderName = "Outputs/Outputs_" + str(datetime.now())
os.mkdir(folderName)
plot = mplot.CorrelationMatrix(folderName+"/KeyMetricsCorrelation.html", "Key Metrics Correlation Matrix", metricNames)
plot.setSeries("Correlation", correlationData)
plot.createOutput()
