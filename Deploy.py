import numpy as np
import math, random

from scipy.spatial import ConvexHull
from scipy.spatial import distance
from matplotlib.path import Path
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch

import Graph

from mclp import mclp

class Deploy():
	"""docstring for Deploy"""
	def __init__(self):
		super(Deploy, self).__init__()

		
		self.range = -1
		self.shortRange = -1 # the radius between the nodes in the same tier
		self.longRange = -1 # the radius between the nodes in different tiers

		self.altitude = None
		self.coverageAngle = 0
		self.maxCoverageRadius = None
		self.edgeCostFunction = None
		self.graph = None
		self.gnList = np.array([])
		self.Tadpole = np.array([[]])
		self.TadpoleNew = np.array([[]])
		self.Froglet = np.array([[]])
		self.cover_nodes_opt = np.array([[]])

		self.areaCentroid = np.array([[]])

	def setRadius(self, diameter):
		# self.range = diameter/2
		self.range = diameter

	def setShortRadius(self, diameter):
		# print('[setShortRadius]')
		# self.shortRange = diameter/2
		self.shortRange = diameter

	def setLongRadius(self, diameter):
		# print('[setLongRadius]')
		# self.longRange = diameter/2
		self.longRange = diameter


	def set_maxOptFroglet(self, K):
		Froglet = self.getFroglet()
		res = mclp(
			candidates = Froglet,
			K = K,
			radius = self.getMaxCost(self.altitude),
			points = self.getPointsOfInterest())
		self.Froglet = res

	def set_maxOptTadpole(self, K):
		Tadpole = self.getTadpole()
		res = mclp(
			candidates = Tadpole,
			K = K,
			radius = self.getMaxCost(self.altitude),
			points = self.getPointsOfInterest())
		self.Tadpole = res

	def avg_distance(self,x,y):
		n = len(x)
		if n <= 1:
			return 1
		dist = 0
		for i in range(n):
			xi = x[i]
			yi = y[i]
			for j in range(i+1,n):
				dx = x[j]-xi
				dy = y[j]-yi
				dist += math.sqrt(dx*dx+dy*dy)
		return 2.0*dist/(n*(n-1))

	def setAltitude(self, val):
		self.altitude = val

	def getAltitude(self):
		return self.altitude

	def setCoverageAngle(self,angle, unit='degrees'):
		if unit == 'radians':
			angle = math.degrees(angle)
		self.coverageAngle = angle
		self.setMaxCoverageRadius()

	def setMaxCoverageRadius(self):
		if not self.altitude:
			print('[setMaxCoverageRadius]','warning: altitude not set')
		else:
			self.maxCoverageRadius = abs(self.altitude*math.tan(self.coverageAngle))

	def calculateEdgeCost(self, nodeA_parameter, nodeB_parameter):
		# print(nodeA_parameter[:-1], nodeB_parameter[:-1])
		# exit(1)
		res = self.edgeCostFunction(nodeA_parameter[:-1], nodeB_parameter[:-1])
		res = 0.01 if res < 0.01 else res
		return res

	def getMaxCost(self, altitude_difference):
		# error: TODO: fix when altitude is too high
		raw_val = 0.0
		if altitude_difference > 0: # if the altitude is too high, the horizontal distance between the center and the coverage node will not be feasible
			if altitude_difference >= self.longRange:
				# print('\t[getMaxCost]','error: altitude of',altitude_difference,'between nodes is too far for the defined MAXLONGRANGE value.','Choose a value between [0, '+str(self.longRange)+'[')
				# return self.longRange
				# exit(1)
				pass
			raw_val = self.longRange
			res = max(0.1,(raw_val**2) - (altitude_difference**2))**0.5
			# print('\t[getMaxCost]: altitude_difference =', altitude_difference, '\tlongRadius:', self.longRange, '\tres:', res)
		else:
			raw_val = self.shortRange
			res = ((raw_val**2) - (altitude_difference**2))**0.5
			# print('\t[getMaxCost]: altitude_difference =', altitude_difference, '\tshortRadius:', self.shortRange, '\tres:', res)
		return res

	def verifyConnection(self, node_a, node_b):
		# print('[verifyConnection]')
		currentDistance = self.calculateEdgeCost(node_a, node_b)
		shortDistance = self.getMaxCost(0)
		# print('\t[verifyConnection]: currentDistance:',currentDistance,'\tshortDistance:', shortDistance, '\t', currentDistance < shortDistance)
		if currentDistance < shortDistance:
			return True
		return False

	def verifyCoverage(self, node_a, node_b):
		currentDistance = self.calculateEdgeCost(node_a, node_b)
		altitude_difference = abs(node_a[-1] - node_b[-1])
		longDistance = self.getMaxCost(altitude_difference)
		# print('\t[verifyCoverage]: currentDistance:',currentDistance,'\tlongDistance:', longDistance, '\t', currentDistance < longDistance)
		if currentDistance < longDistance:
			return True
		return False


	def setCostCalculation(self, func):
		self.edgeCostFunction = func

	def getMaxCoverageRadius(self):
		return self.maxCoverageRadius

	def pushCoveragePosition(self, position):
		# position[0][-1] = self.getAltitude()
		if len(self.Tadpole[0]) < 1:
			self.Tadpole = np.array(position)
		else:
			self.Tadpole = np.append(self.Tadpole, position, axis=0)

		# if len(self.gnList) <= len(self.getTadpole()):
		# 	exit()


	def setPointsOfInterest(self, gnList):
		newList = []
		if gnList.shape[1] == 1:
			print('[setPointsOfInterest]','error: gnList in wrong shape',gnList.shape)
			exit(1)
		elif gnList.shape[1] == 2:
			for row in gnList:
				newList.append(np.append(row, 1.0))
			newList = np.array(newList)
			self.gnList = newList
		else:
			self.gnList = gnList

	def getPointsOfInterest(self):
		if len(self.gnList) < 1:
			print('[getPointsOfInterest]','error: gnList not defined')
			exit(1)
		else:
			return self.gnList


	def resultantVector(self,origin,pointA,pointB,dist):
		res = 0
		vAB = np.array([origin,pointA])
		vAC = np.array([origin,pointB])
		vec = vAB + vAC - origin

		vec_unit_length = 1/self.calculateEdgeCost(vec[0],vec[1])

		vec_length = vec_unit_length * dist
		# ====== Edge/adjust factor ====
		# It prevents the user to be too close to the communication range limit of a UAV.
		# It puts the center of the UAV closer to the user if closer to -1. # [-1,0[
		# It puts the center of the UAV far from the user if closer to 1. # ]0,1]
		
		avgdist = (self.calculateEdgeCost(origin,pointA)+self.calculateEdgeCost(origin,pointB) )/2
		
		# print(f'\u001b[33mdist: {dist}\u001b[0m')
		# print(f'\u001b[33mavgdist: {avgdist}\u001b[0m')
		# print(f'\u001b[33m{avgdist/dist}\u001b[0m')
		# print(f'\u001b[33m{dist/avgdist}\u001b[0m')
		factor = (1 - ((dist/avgdist)**(1/3)))
		# print(f'\u001b[33mfactor: {factor}\u001b[0m')
		# vec_length *= factor

		res = (1-vec_length)*vec[0] + vec_length * vec[1];

		return res

	def getConvexHullPoints(self, points3D):
		points2D = points3D[:,0:2]
		hull = ConvexHull(points2D)
		hull_path = Path(points2D[hull.vertices])
		res = np.array([points2D[p] for p in hull.vertices])
		mask = np.isin(points2D,res)
		return points3D[mask[:,0]]

		# subset = np.array(res)
		# return subset

	def midpoint(self, p1, p2):
		return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
	
	# def getFarthestPoint(self, point, arrPoints):
	def getFarthestPoint(self, *args, arrPoints=None):
		if len(args) == 1: #only one point in the input argument
			point = args[0]

		elif len(args) == 2: #two points in the input argument
			# print(args)
			# print()
			# input("len(args) == 2")
			point = self.midpoint(args[0], args[1])

		else:
			print('[getFarthestPoint]','error: invalid number of arguments')
			print(len(args), args)
			exit(1)

		maxDist = 0
		res = point
		for i in arrPoints:
			currentDistance = self.calculateEdgeCost(point, i)
			# print('\tdistance compare 1:', maxDist,currentDistance)
			if maxDist < currentDistance:
				# print('distance:', currentDistance)
				res = i
				maxDist = currentDistance
			# print('\tdistance compare 2:', maxDist,currentDistance)
			# input('\tdistance compare...')

		return res

	def getTadpole(self):
		if len(self.Tadpole[0]) < 1:
			print('[getTadpole]','error: Tadpole not defined')
			return None
		return self.Tadpole

	def getTadpoleNew(self):
		if len(self.TadpoleNew) < 1:
			# print('[getTadpoleNew]','error: TadpoleNew not defined')
			return self.TadpoleNew
		return self.TadpoleNew

	def set_cover(self, universe, subsets):
		# http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
		"""Find a family of subsets that covers the universal set"""
		elements = set(e for s in subsets for e in s)
		
		# Check the subsets cover the universe
		if elements != universe:
			print('[set_cover]','error: subsets not covering the universe')
			return '[set_cover] error: subsets not covering the universe',None
		covered = set()
		cover = []

		# Greedily add the subsets with the most uncovered points
		while covered != elements:
			subset = max(subsets, key=lambda s: len(s - covered))
			cover.append(subset)
			covered |= subset
		return cover

	def optimizeTadpole(self):
		# print('[optimizeTadpole]')
		if len(self.Tadpole) == 1:
			# move node to centroid
			self.Tadpole[0][:2] = self.areaCentroid[:2]
			# print(f'self.areaCentroid: {self.areaCentroid}')
			# print(f'self.Tadpole[0]: {self.Tadpole[0]}')
			# print(f'self.Tadpole[0]: {self.Tadpole[0]}')
			# exit()



		setNameList = [{idSet:aSet} for idSet, aSet in enumerate(self.Tadpole)]
		elementNameList = [{idElement:element} for idElement, element in enumerate(self.gnList)]

		subsetList = []
		refSubsetList = []
		universe = set()
		
		# Find the elements covered by each tadpole node
		# build the subsets of covered elements {1,2,3}, {1,2}, {4,5,9}, ...
		for ids, setname in enumerate(setNameList):
			# print(ids, 'covers')
			elementList = []
			for ide, element in enumerate(elementNameList):
				universe |= set([ide])
				# if self.calculateEdgeCost(setname[ids], element[ide]) < self.cost:
				if self.verifyCoverage(setname[ids], element[ide]):
					# print('\t',ide)
					elementList.append(ide)
			refSubsetList.append({ids:set(elementList)})
			subsetList.append(set(elementList))

		self.cover_nodes_opt = self.Tadpole
		final_subset = self.set_cover(universe, subsetList)

		nodeIds = []
		for i,rfl in enumerate(refSubsetList):
			for fss in final_subset:
				if rfl[i] == fss:
					nodeIds.append(i)

		self.Tadpole = self.Tadpole[nodeIds]

	def plotNumpy3D(self, arr, color=None, time=None, size=None):
		plt.clf()
		w, h = figaspect(1)
		w *= 1.6
		h = w

		fig = plt.figure(figsize=(w,h))
		ax = fig.add_subplot(111, projection='3d')
		ax.view_init(azim=100, elev=60)
		# ax = fig.add_subplot(111) # not 3D

		ax.grid(linestyle='--')
		ax.set_axisbelow(True)

		ax.tick_params(axis='both',which='both',left=False,top=False,right=False,bottom=False)
		x = arr[:,0]
		y = arr[:,1]
		z = arr[:,2]

		ax.scatter(x,y,z,c=('C0' if color is None else color), s=(10 if size is None else size))
		ax.scatter(self.gnList[:,0],self.gnList[:,1],self.gnList[:,2],c='green', s=50, alpha=0.3)

		# ax.scatter(x,y,c=('C0' if color is None else color), s=100)# not 3D
		# ax.scatter(self.gnList[:,0],self.gnList[:,1],c='green', s=50, alpha=0.2)# not 3D


		for i in arr:
			realRadiusRange = ((((self.longRange)**2)+(z[0]**2))**0.5)
			# p = Circle((i[0],i[1]), self.longRange, color=color, alpha=0.3, lw=0.8)
			p = Circle((i[0],i[1]), realRadiusRange, color=color, alpha=0.3, lw=0.8)
			ax.add_patch(p)
			art3d.pathpatch_2d_to_3d(p, z=1.5, zdir="z")


		# ax.set_xlim(-10,120)
		# ax.set_ylim(-10,120)

		if time is not None:
			if time > 0:
				plt.pause(time)
				plt.clf()
		else:
			plt.show()
		# print()

	def calculateTadpole(self):
		# print('altitude:', self.altitude)
		# print('cost:',self.cost)
		self.Tadpole = np.array([[]])
		UElocatioins = self.gnList


		# cent = np.mean(UElocatioins[:,-2:], axis=0)
		cent = np.mean(UElocatioins[0:,:3], axis=0)
		self.areaCentroid = cent

		if len(UElocatioins) == 2:
			cent[-1] = self.altitude
			self.pushCoveragePosition([cent])
			# self.optimizeTadpole()
			return


		if len(UElocatioins) == 1:
			newNode = UElocatioins[0]
			newNode[-1] = self.altitude
			self.pushCoveragePosition([newNode])
			# self.optimizeTadpole()
			return

		selectedMemory = []

		while len(UElocatioins) > 0:


			if len(UElocatioins) == 2:
				pUE = self.resultantVector(UElocatioins[0], UElocatioins[1], cent, self.getMaxCost(abs(UElocatioins[0][-1]-self.altitude)) - 1);
				pUE[-1] = self.altitude
				self.pushCoveragePosition([pUE])
				UElocatioins = np.array([i for i in UElocatioins if not self.verifyCoverage(i, pUE)])

				if len(UElocatioins) == 1:
					pUE = self.resultantVector(UElocatioins[0], cent, cent, self.getMaxCost(abs(UElocatioins[0][-1]-self.altitude)) - 1);
					pUE[-1] = self.altitude
					self.pushCoveragePosition([pUE])
					UElocatioins = np.array([i for i in UElocatioins if not self.verifyCoverage(i, pUE)])
					break
				break

			if len(UElocatioins) == 1:
				pUE = self.resultantVector(UElocatioins[0], cent, cent, self.getMaxCost(abs(UElocatioins[0][-1]-self.altitude)) - 1);
				pUE[-1] = self.altitude
				self.pushCoveragePosition([pUE])
				UElocatioins = np.array([i for i in UElocatioins if not self.verifyCoverage(i, pUE)])
				# self.plotNumpy3D(self.Tadpole, color='blue')
				break

			if len(UElocatioins) >= 3:
				# cent = np.mean(UElocatioins[:,-2:], axis=0)
				cent = np.mean(UElocatioins[0:,:3], axis=0)

			ConvexHull_points = self.getConvexHullPoints(UElocatioins)

			# % Choosing A, B, C
			A = ConvexHull_points[0]
			# print('this is A',A)
			B = self.getFarthestPoint(A, arrPoints=ConvexHull_points)
			# print('this is B',B)
			C = self.getFarthestPoint(A, B, arrPoints=ConvexHull_points)
			# print('this is C',C)

			# Place a coverage disk over and tangenting A, B, and C
			# for A to B and C
			selectedUserNodeList = [A,B,C]
			for idx, selectedUserNode in enumerate(selectedUserNodeList):
				nA = selectedUserNodeList[(idx+0)%len(selectedUserNodeList)]
				nB = selectedUserNodeList[(idx+1)%len(selectedUserNodeList)]
				nC = selectedUserNodeList[(idx+2)%len(selectedUserNodeList)]
				newPoint = np.array([self.resultantVector(nA, nB, nC, self.getMaxCost(abs(A[-1]-self.altitude))-1)]); # get resultant point from vectors AB and AC
				newPoint[0][-1] = self.altitude

				UElocatioins = [i for i in UElocatioins if not self.verifyCoverage(i, newPoint[0])] # remove all covered points
				self.pushCoveragePosition(newPoint)
				if len(UElocatioins) < 1:
					break


			UElocatioins = np.array(UElocatioins)
			
			# self.plotNumpy3D(self.Tadpole, color='blue', time=2)
			# exit()
		# self.plotNumpy3D(self.Tadpole, color='green', time=2)
		if len(self.Tadpole) > 1:
			self.optimizeTadpole()
		# self.plotNumpy3D(self.Tadpole, color='red', time=2)


	def isNeighbours(self, nodeA, nodeB, subgraphs):
		for sg in subgraphs:
			if nodeA in list(sg) and nodeB in list(sg): # check if nodes are in the same subgraph
				return True
				break
		return False

	def calculateFroglet(self):
		# print(self.Tadpole)
		if len(self.Tadpole) < 1:
			print('[calculateFroglet]','error: Tadpole not defined')
			exit(1)
		else:
			if len(self.Tadpole[0]) < 1:
				print('[calculateFroglet]','error: Tadpole not defined')
				exit(1)

		# build pair edges based on the cost between each pair			
		gf = Graph
		gf.set_connectivetyparameter(self.getMaxCost(0), self.calculateEdgeCost)
		gf.build_graph(self.getTadpole())

		# find the number of disconnected graphs
		subgraphs = gf.get_subgraphs()
		inverse_graph = gf.get_inverse_graph()
		inverse_subgraphs = gf.get_inverse_subgraphs()

		newEdgesList = []
		joinSubgraphs = set({})
		inverseGraphEdgesList = sorted(inverse_graph.edges(data=True), key=lambda t: t[2].get('weight', 1))
		# exit()


		newEdgesList = []
		for pair in inverseGraphEdgesList:
			if not self.isNeighbours(pair[0], pair[1], subgraphs):
				newEdgesList.append(pair)
				sub1 = set({})
				sub2 = set({})
				for sg in subgraphs:
					if pair[0] in list(sg) or pair[1] in list(sg):
						sub1 = sub1.union(sg)
						subgraphs.remove(sg)
				for sg in subgraphs:
					if pair[0] in list(sg) or pair[1] in list(sg):
						sub2 = sub2.union(sg)
						subgraphs.remove(sg)

				if sub1 != sub2:
					joinSubgraphs = sub1.union(sub2)
					subgraphs.append(joinSubgraphs)
				else:
					subgraphs.append(sub1)

		# exit()

		# place new nodes between the disconnected graphs
		self.graph = gf.get_graph()
		newNodeID = list(self.graph.nodes)[-1]+1
		self.Froglet = self.Tadpole

		newNodesList = []

		# gf.draw(self.graph)
		for pair in newEdgesList:
			# calculate the center between each pair
			# get the position of the pair 1
			pos1 = self.getTadpole()[pair[0]]
			# get the position of the pair 2
			pos2 = self.getTadpole()[pair[1]]

			costBetween = self.calculateEdgeCost(pos1,pos2)
			numberNewNodes = math.ceil((costBetween-(self.getMaxCost(0)*2))/(self.getMaxCost(0)*2))
			
			# calculate the middleNodes
			for i in range(1,numberNewNodes+1):

				# splitting length between edges into smaller lengths
				t = (1/(1+numberNewNodes))*i
				middleNodes = [pos1[0] + (pos2[0]-pos1[0]) * t, pos1[1] + (pos2[1]-pos1[1]) * t]
				middleNodes.append(pos1[2])

				self.Froglet = np.append(self.Froglet,[middleNodes], axis=0)
				# place a new node between the each pair
				self.graph.add_nodes_from({newNodeID:middleNodes})
				gf.appendPos(newNodeID, middleNodes[:-1]) # for print purpose. do not remove
				newNodesList.append(middleNodes)

				# add the edges among them all
				self.graph.add_edge(*(pair[0],newNodeID), weight=self.calculateEdgeCost(middleNodes, pos1))
				self.graph.add_edge(*(pair[1],newNodeID), weight=self.calculateEdgeCost(middleNodes, pos2))

				newNodeID += 1

				#self.plotNumpy3D(self.Froglet, color='purple', time=0.5)

		self.TadpoleNew = np.array(newNodesList)

	def getFroglet(self):
		if len(self.Froglet) < 1:
			print('[getFroglet]','error: Froglet not defined')
			return None
		return self.Froglet


