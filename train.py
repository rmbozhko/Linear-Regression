#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

thetas = None 

def 	featureScaling(data):
	max_value = np.amax(data)
	return (np.divide(data, max_value))

def 	meanNormalization(data):
	return (np.divide((data - np.mean(data)), np.std(data)))

def 	computeCostSGD(X, Y):
	global thetas

	J = h_function(X)
	J = np.sum(np.square(J - Y) / 2) / (Y.shape[0])
	return (J)

def 	computeCostBGD(X, Y):
	global thetas

	J = h_function(X)
	J = np.sum(np.square(J - Y)) / (2 * Y.shape[0])
	return (J)

def 	h_function(X):
	global thetas
	return (X.dot(thetas))

def	normalEquation(X, Y):
	global thetas
	
	X = addBiasUnit(X)
	thetas = np.array(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y))

def 	SGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=1500):
	global thetas

	# in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
	m = Y.shape[0]
	
	# Metrics storages
	thetasHistory = list()
	iterations = list()

	for j in range(iterationsNum):
		for i in range(X.shape[0]):
			X_temp = np.array([X[i, :]])
			J = (h_function(X_temp) - Y[i]).dot(X_temp)
			J = learningRate * J
			thetas = thetas - J
		# Metrics collecting
		thetasHistory.append(computeCost(X, Y))
		iterations.append(j)

	return (iterations, thetasHistory)

def	BGD(X, Y, computeCost, h_function, learningRate=0.0001, iterationsNum=1500):
	global thetas

	# in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
	m = Y.shape[0]
	
	# Metrics storages
	thetasHistory = list()
	iterations = list()

	for i in range(iterationsNum):
		J = (h_function(X) - Y).dot(X)
		J = learningRate * np.divide(J, m)
		thetas = thetas - J
		
		# Metrics collecting
		thetasHistory.append(computeCost(X, Y))
		iterations.append(i)

	return (iterations, thetasHistory)

def	addBiasUnit(arr):
	bias_arr = np.ones((arr.shape[0], 1), dtype=float)
	return (np.column_stack((bias_arr, arr)))

def	calcAccuracy(X, Y, logReg=True):
	global thetas

	if logReg:
		temp_y = X.dot(thetas)
		pred = np.mean(Y == temp_y) * 100
	else:
		pred = int(np.sum(Y - X.dot(thetas)))
	return (pred)

def	computeThetas(X, y, gradDesc, h_func, computeCost):	
	# adding bias column to X data
	X = addBiasUnit(X)
	
	# cycle vars
	diff = 1
	learningRate = 1.0
	step = 0.1
	
	# determing best-fitting learningRate using brute-force	
	while diff > 0.00000001:
		for i in range(9):
			if diff <= 0.0001 and diff >= 0:
				break
			learningRate = learningRate - step
			[iterations, history] = gradDesc(X, y, computeCost, h_func, learningRate)
			diff = calcAccuracy(X, y, False)
			print("diff:{}".format(diff))
		step = step * 0.1
	print("learningRate:{}".format(learningRate))
	return ([history, iterations])

def	displayGraph(X, Y):
	global thetas

	if X.shape[1] == 1:
		# plotting 2D graph, with prediction line
		plt.figure(1)	
		data, = plt.plot(X, Y, 'bo', label='Training data')
		dummy_x = np.linspace(np.min(X), np.max(X), 100)
		plt.plot(dummy_x, thetas[0] + (dummy_x * thetas[1]), 'r')
		plt.ylabel(u'X\u2081')
		plt.xlabel(u'X\u2082')	
		plt.legend()
		plt.show()

	elif X.shape[1] == 2:
		# plotting 3D graph, with prediction line
		fig = plt.figure(2)
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:, 0], X[:, 1], Y, c='b', marker='o')
		temp_x0 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
		temp_x1 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
		temp_y = np.array([thetas[0] + (thetas[1] * temp_x0), thetas[0] + (thetas[2] * temp_x1)])
		ax.plot_surface(temp_x0, temp_x1, temp_y, rstride=4, cstride=4, alpha=0.8, cmap='Reds')
		ax.set_xlabel(u'X\u2081')	
		ax.set_ylabel(u'X\u2082')	
		ax.set_zlabel('Y')
		plt.legend()
		plt.show()

def     main(dataset):
	global thetas
	# reading data from a file, except header line
	data = np.genfromtxt(dataset, delimiter=',', skip_header=1)

	Y = data[:, -1]
	X = data[:, :-1]
	X_old = data[:, :-1]
	
	thetas = np.zeros(X.shape[1] + 1)
	
	# normalizing features
	if args.is_fscale:
		X = featureScaling(X)
	else:
		X = meanNormalization(X)

	# Computing thetas
	if args.is_norm:
		normalEquation(X, Y)
	else:
		if args.is_sgd:
			[history, iterations] = computeThetas(X, Y, SGD, h_function, computeCostSGD)
		else:
			[history, iterations] = computeThetas(X, Y, BGD, h_function, computeCostBGD)
			# plotting cost function results
		plt.plot(iterations, history)
		plt.ylabel('Function cost')
		plt.xlabel('Iterations')
		plt.show()
	
	displayGraph(X, Y)

	print("Difference between calculated and passed output values: {}".format(calcAccuracy(addBiasUnit(X), Y, False)))
	
	# saving thetas to temp file
	with open('thetas.txt', 'w') as f:
		for i in range(thetas.shape[0]):
			f.write(str(thetas[i]) + "\n")
	
	# saving metrics to temp file
	with open('metrics.txt', 'w') as f:
		for i in range(X_old.shape[1]):
			f.write(str(np.max(X_old[:, i])) + " " + str(np.mean(X_old[:, i])) + " " + str(np.std(X_old[:, i])) + "\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train thetas for further precition.')
	parser.add_argument('-norm', dest='is_norm', action='store_true',         default=False, help='choose normal equation as thetas training algorithm')
	parser.add_argument('-bgd', dest='is_bgd', action='store_true',         default=True, help=' [default] choose batch gradient descent as thetas training algorithm')
	parser.add_argument('-sgd', dest='is_sgd', action='store_true',         default=False, help='choose stohastic gradient descent as thetas training algorithm')
	parser.add_argument('-meanNorm', dest='is_fscale', action='store_false', default=True, help='choose mean normalization as method to rescale input data')
	parser.add_argument('-fscale', dest='is_fscale', action='store_true', default=True, help=' [default] choose feature scalling as method to rescale input data')
	requiredArgs = parser.add_argument_group('Required arguments')
	requiredArgs.add_argument('-data', help='dataset with input values to train thetas', required=True)
	args = parser.parse_args()
	main(args.data)
