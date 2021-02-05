#!/usr/bin/env python3

# --------------------- REFERENCE ---------------------
# https://en.wikipedia.org/wiki/Maximum_coverage_problem#:~:text=The%20maximum%20coverage%20problem%20is,widely%20taught%20in%20approximation%20algorithms.&text=of%20these%20sets%20such%20that,selected%20sets%20has%20maximal%20size.
# https://github.com/cyang-kth/maximum-coverage-location
# -----------------------------------------------------

import pulp as plp
import random
from scipy.spatial import distance_matrix
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt


def generate_candidate(points,M=100):
	'''
	Generate M candidate candidates with the convex hull of a point set
	Input:
		points: a Numpy array with shape of (N,2)
		M: the number of candidate candidates to generate
	Return:
		candidates: a Numpy array with shape of (M,2)
	'''
	hull = ConvexHull(points)
	polygon_points = points[hull.vertices]
	poly = Polygon(polygon_points)
	min_x, min_y, max_x, max_y = poly.bounds
	candidates = []
	while len(candidates) < M:
		random_point = Point([random.uniform(min_x, max_x),
							 random.uniform(min_y, max_y)])
		if (random_point.within(poly)):
			candidates.append(random_point)
	return np.array([(p.x,p.y) for p in candidates])

def plot_input(points):
	'''
	Plot the result
	Input:
		points: input points, Numpy array in shape of [N,2]
		opt_candidates: locations K optimal candidates, Numpy array in shape of [K,2]
		radius: the radius of circle
	'''
	fig = plt.figure(figsize=(8,8))
	plt.scatter(points[:,0],points[:,1],c='C1')
	ax = plt.gca()
	ax.axis('equal')
	ax.tick_params(axis='both',left=False, top=False, right=False,
					   bottom=False, labelleft=False, labeltop=False,
					   labelright=False, labelbottom=False)
	ax.grid(linestyle='--')
	ax.set_axisbelow(True)

def plot_result(points,opt_candidates,radius):
	'''
	Plot the result
	Input:
		points: input points, Numpy array in shape of [N,2]
		opt_candidates: locations K optimal candidates, Numpy array in shape of [K,2]
		radius: the radius of circle
	'''
	fig = plt.figure(figsize=(8,8))
	plt.scatter(points[:,0],points[:,1],c='C0')
	ax = plt.gca()
	plt.scatter(opt_candidates[:,0],opt_candidates[:,1],c='C2',marker='x',s=75)
	for site in opt_candidates:
		circle = plt.Circle(site, radius, color='C2',fill=False,lw=1.5)
		ax.add_artist(circle)
	ax.axis('equal')
	ax.tick_params(axis='both',left=True, top=True, right=True,
					   bottom=True, labelleft=True, labeltop=True,
					   labelright=True, labelbottom=True)
	ax.grid(linestyle='--')
	ax.set_axisbelow(True)
	ax.grid(linestyle='--')

	plt.show()


def mclp(candidates=None, K=None, radius=None, points=None):
	K = int(K)
	J = candidates.shape[0]
	I = points.shape[0]
	D = distance_matrix(points,candidates)

	mask = D <= radius
	D[mask] = 1
	D[~mask] = 0

	y_vars = {(i):plp.LpVariable(cat=plp.LpBinary, name="y_%d"%i) for i in range(I)}
	x_vars = {(j):plp.LpVariable(cat=plp.LpBinary, name="x_%d"%j) for j in range(J)}

	opt_model = plp.LpProblem()
	opt_model.addConstraint(plp.LpConstraint(
		e = plp.lpSum(x_vars[j] for j in range(J)),
		sense = plp.LpConstraintEQ,
		rhs=K,
		name="constraint_%d"%0))

	for i in range(I):
		mask = np.where(D[i] == 1)[0]
		if len(mask) > 0:
			newConstraint = plp.LpConstraint(
				e = plp.lpSum(x_vars[j]-y_vars[i] for j in mask),
				sense = plp.LpConstraintGE,
				rhs = 0,
				name = "constraint_%d"%len(opt_model.constraints))
			opt_model.addConstraint(newConstraint)

	objective = plp.lpSum(y_vars[i] for i in range(I))
	opt_model.sense = plp.LpMaximize
	opt_model.setObjective(objective)

	opt_model.solve(plp.PULP_CBC_CMD(msg=0))
	# opt_model.solve()

	solution = []
	if opt_model.status == 1:
		for v in opt_model._variables:
			if v.varValue == 1 and v.name[0] == 'x':
				solution.append(int(v.name[2:]))
	opt_candidates = candidates[solution]

	return opt_candidates

# # ----------------------- user input -----------------------
# points = np.array([[65.91018903,98.62297763,1.5],[50.39484214,60.9712284,1.5],[76.44405577,47.37046773,1.5],[33.27374414,31.23787994,1.5],[85.29110873,73.44537979,1.5],[68.11354958,55.45892658,1.5],[-1.06917666,103.34926562,1.5],[63.94574005,112.52024089,1.5],[32.1175471,58.5619233,1.5],[83.14060132,49.72293825,1.5]])
# # candidates = np.array([[71.73577816222846,53.69579652138079,6.5],[62.68257554158404,104.73679192597434,6.5],[38.99287023951905,62.42295866348836,6.5]])
# candidates = np.array([
# 	[62.68257554158404,104.73679192597434,6.5],
# 	# [34.478545708723715,39.03057623402523,6.5],
# 	[71.73577816222846,53.69579652138079,6.5],
# 	# [4.953246368944354,98.2592672691796,6.5],
# 	# [70.05969407233393,88.74191014948828,6.5],
# 	[38.99287023951905,62.42295866348836,6.5],
# 	[55.36432420087375,58.05937759243457,6.5],
# 	# [77.43681260308381,72.74702837300221,6.5],
# 	[21.9730583042317,80.34111296633398,6.5]])
# # np.array([[77.43681260308381,72.74702837300221,6.5],
# # 	[62.68257554158404,104.73679192597434,6.5],
# # 	[34.478545708723715,39.03057623402523,6.5],
# # 	[71.73577816222846,53.69579652138079,6.5],
# # 	[11.73577816222846,13.69579652138079,6.5],
# # 	[101.73577816222846,23.69579652138079,6.5],
# # 	[28.73577816222846,203.69579652138079,6.5],
# # 	[4.953246368944354,98.2592672691796,6.5],
# # 	[70.05969407233393,88.74191014948828,6.5],
# # 	[55.36432420087375,58.05937759243457,6.5],
# # 	[38.99287023951905,62.42295866348836,6.5],
# # 	[21.9730583042317,80.34111296633398,6.5]])
# radius = 30
# K = len(candidates)/2
# # ----------------------------------------------------------
# res = mclp(candidates=candidates, K=K, radius=radius, points=points)

# # plot_input(points)
# plot_result(points, res, radius)
# plt.show()
