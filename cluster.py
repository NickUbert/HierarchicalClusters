#Credit to Imran Ahmed
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np


#Steps:
# 1.) Create a seperate cluster for each data point
# 2.) Group only those points that are closest to each other
# 3.) We check for the stop condition, if not repeat 2.

#Create dataframe
dataset = pd.DataFrame({
    'x': [11, 11, 20, 12, 16, 33, 24, 14, 45, 52, 51, 52, 55, 53, 55, 61, 62, 70, 72, 10],
    'y': [39, 36, 30, 52, 53, 46, 55, 59, 12, 15, 16, 18, 11, 23, 14, 8, 18, 7, 24, 70]
})

#Create Clusters
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(dataset)

#Results
print(cluster.labels_)  

#Evaluating Results:
#Range , Meaning , Description
# .71 - 1.0, Excellent, Groups are very differentiable
# .51 - .70, Reasonable, Groups are somewhat differentiable
# .26 - .50, Weak, Quality of groups should not be relied upon
# <.26, No clustering, it was not possible to create cluster groups