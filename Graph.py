#!/usr/bin/env python3
import os
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance


GRAPH = None
SUBGRAPH = None
INVERSEGRAPH = None
INVERSESUBGRAPH = None
POS = {}
DISTANCE_RANGE = None
COSTFUNC = None

# The usual parameter that establish connectivity is the distance range, however,
# in complex environment, the nodes mi suffer with interference, blockage, etc...
# the parameter shall be a broadcast ping to other nodes and see who responds


def set_connectivetyparameter(val, func):
	global DISTANCE_RANGE
	global COSTFUNC

	# DISTANCE_RANGE = val*2
	DISTANCE_RANGE = val
	COSTFUNC = func

def get_connectivetyparameter():
	return DISTANCE_RANGE

def get_graph():
	global GRAPH
	return GRAPH

def get_subgraphs():
	global SUBGRAPH
	return SUBGRAPH

def get_inverse_graph():
	global INVERSEGRAPH
	return INVERSEGRAPH

def get_inverse_subgraphs():
	global INVERSESUBGRAPH
	return INVERSESUBGRAPH

def appendPos(key, pos):
	global POS
	POS[key] = pos

def build_graph(listOfNodes):
	global GRAPH
	global SUBGRAPH
	global INVERSEGRAPH
	global INVERSESUBGRAPH

	GRAPH = nx.Graph()
	INVERSEGRAPH = nx.Graph()

	first_name_ascii = 65

	completedic = {}
	for i,val in enumerate(listOfNodes):
		completedic[i] = val
		
		# GRAPH.add_node(chr(i+first_name_ascii))
		GRAPH.add_node(i)
		INVERSEGRAPH.add_node(i)

		# POS[chr(i+first_name_ascii)] = val[0:-1]
		POS[i] = val[0:-1]

	templist = completedic
	listDistances = {}
	for i in completedic:
		templist = {t:templist[t] for t in templist if t!=i}
		vals = []
		for j in templist:
			if is_connected(completedic[i],templist[j]):
				# GRAPH.add_edge(*(chr(i+first_name_ascii),chr(j+first_name_ascii)))

				GRAPH.add_edge(*(i,j),weight=COSTFUNC(completedic[i],templist[j]))
			else:
				INVERSEGRAPH.add_edge(*(i,j),weight=COSTFUNC(completedic[i],templist[j]))

	SUBGRAPH = list(nx.connected_components(GRAPH))
	INVERSESUBGRAPH = list(nx.connected_components(INVERSEGRAPH))


def is_connected(a,b):
	return COSTFUNC(a,b) < get_connectivetyparameter()

def draw(G):
	nx.draw(G, POS,node_size = 400, with_labels = True, alpha=0.6)
