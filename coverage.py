#!/usr/bin/env python3
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from matplotlib.path import Path

from sklearn.cluster import MiniBatchKMeans

class Coverage():
	"""docstring for Coverage"""
	def __init__(self, costfunction, groundNodes, skyNodes):
		super(Coverage, self).__init__()
		self.costfunction = costfunction
		self.groundNodes = groundNodes
		self.skyNodes = skyNodes






def get_clusters(posarr):
	# print(posarr)
	# n_clusters = int(len(posarr)**(1/2.5))
	n_clusters = 3
	# kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6)
	kmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=len(posarr), n_init=1000, max_no_improvement=1000, verbose=0)

	steps = np.linspace(-1,len(posarr),n_clusters)
	for i,val in enumerate(steps):
		if (i+1) == len(steps):
			break
		kmeans = kmeans.partial_fit(posarr[int(steps[i])+1:int(steps[i+1]),:])

	kmeans.fit(posarr)

	clusters = kmeans.cluster_centers_
	return clusters