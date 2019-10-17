#!/usr/bin/env python3.7.4
# -*- Coding: UTF-8 -*-

"""
Hierarchical and non-hierarchical clustering
"""

class cluster():
	"""
	Create clusters to classify data
	"""

	def sanitize(self, df):
		"""
		Normalize data and select columns
		"""
		maximum = max(df['var0'].values)
		df['var0'] = df['var0'].values/maximum
		data = df.loc[:, ['var0', 'var1']].values
		return(df, data, maximum)

	def color_map(self, method):
		"""
		Create color map with correspondence between place on graphic
		and value/color
		"""
		# Create mapcolors for 4 values
		from matplotlib import pyplot as plt
		import matplotlib.colors
		cvals  = [0, 1, 2, 3]
		# Correlation between place on graphic and value/color
		if method == 'ward':
			colors = ['black', 'gray', 'blue', 'yellow']
		elif method == 'kmeans':
			colors = ['yellow', 'black', 'blue', 'gray']
		norm = plt.Normalize(min(cvals),max(cvals))
		tuples = list(zip(map(norm,cvals), colors))
		cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
		return(cmap)

	def ward_dendrogram(self, data):
		"""
		Calculate and plot dendogram using Ward's method
		(Hierarchical clustering)
		"""
		from matplotlib import pyplot as plt
		import scipy.cluster.hierarchy as sch
		# Plot dendrogram
		plt.figure(figsize=(10, 7))
		plt.title('Dendrogram')
		dend = sch.dendrogram(sch.linkage(data, method='ward'))
		plt.savefig('dendrogram.png')
		plt.close()
		return()
		
	def ward_cluster(self, data, n_clusters, maximum):
		"""
		Calculate hierarquical clusters (Ward's method) and plot
		"""
		# Make clusters
		from sklearn.cluster import AgglomerativeClustering
		cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
		cluster.fit_predict(data)
		
		# Plot
		from matplotlib import pyplot as plt
		plt.figure(figsize=(10, 7))
		# Define color map
		#cmap = 'jet'
		cmap = self.color_map('ward')
		# De-normalize before plot
		plt.scatter(data[:,0]*maximum, data[:,1], c=cluster.labels_, cmap=cmap)
		plt.title('Clusters - Ward')
		plt.xlabel('var0')
		plt.ylabel('var1')
		plt.savefig('clusters_ward.png')
		plt.close()
		return()

	def kmeans_elbow(self, data):
		"""
		Elbow's method to estimate best number of clusters
		"""
		from math import sqrt
		from matplotlib import pyplot as plt
		from sklearn.cluster import KMeans
		# Calculate sum of squares
		wcss = []
		for n in range(2, 21):
			kmeans = KMeans(n_clusters=n)
			kmeans.fit(X=data)
			wcss.append(kmeans.inertia_)
		# Calculate best number of clusters
		x1, y1 = 2, wcss[0]
		x2, y2 = 20, wcss[len(wcss)-1]
		distances = []
		for i in range(len(wcss)):
			x0 = i+2
			y0 = wcss[i]
			numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
			denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
			distances.append(numerator/denominator)
		n = distances.index(max(distances)) + 2
		print(n)
		return()

	def kmeans_cluster(self, data, n_clusters, maximum):
		"""
		Calculate K-means and plot
		"""
		# Make clusters
		from sklearn.cluster import k_means
		import numpy as np
		init = np.array([[1.25,0.28], [4.29,0.71], [0.99,1.15], [6.5,1.011]])
		centroids, labels, sse = k_means(data, n_clusters=n_clusters, init=init, n_init=100)
		# Plot
		from matplotlib import pyplot as plt
		plt.figure(figsize=(10, 7))
		# Define color map
		#cmap = 'jet'
		cmap = self.color_map('kmeans')
		# De-normalize before plot
		plt.scatter(data[:,0]*maximum, data[:,1], c=labels, cmap=cmap)
		plt.title('Clusters - k-means')
		plt.xlabel('var0')
		plt.ylabel('var1')
		plt.savefig('clusters_kmeans.png')
		plt.close()
		return()
