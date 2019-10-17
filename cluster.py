# Hierarchical and non-hierarchical clustering

import sys
sys.path.append('/home/vinicius/Documentos')
import functions
cluster = functions.cluster()
import pandas as pd

n_clusters = 4
df = pd.read_csv('example.csv')
df, data, maximum = cluster.sanitize(df)

cluster.ward_dendogram(data)
cluster.ward_cluster(data, n_clusters, maximum)
#cluster.kmeans_elbow(data)
#cluster.kmeans_cluster(data, n_clusters, maximum)
