#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

thetas = None 

def 	featureScaling(data):
	max_value = np.amax(data)
	return (np.divide(data, max_value))

def 	meanNormalization(data):
	return (np.divide((data - np.mean(data)), np.std(data)))

def 	computeCost(X, Y):
	global thetas

	J = h_function(X)
	J = np.sum(np.square(J - Y)) / (2 * Y.shape[0])
	return (J)

def 	h_function(X):
	global thetas
	return (X.dot(thetas))

#def	normalEquation(X, Y):
#	thetas = np.zeros(2)

#	thetas[1] = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
#	print(thetas)

def 	gradientDescent(X, Y, learningRate=0.0001, iterationsNum=1500):
	global thetas

	# in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
	m = Y.shape[0]
	
	# Metrics storages
	thetasHistory = list()
	iterations = list()
	thetasZeroStrg = list()
	thetasOneStrg = list()

	for i in range(iterationsNum):
		J = (h_function(X) - Y).dot(X)
		J = learningRate * np.divide(J, m)
		thetas = thetas - J
		
		# Metrics collecting
		thetasZeroStrg.append(thetas[0])
		thetasOneStrg.append(thetas[1])
		thetasHistory.append(computeCost(X, Y))
		iterations.append(i)
	return (iterations, thetasHistory, thetasZeroStrg, thetasOneStrg)

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

def	computeThetas(X, y):	
	# adding bias column to X data
	X = addBiasUnit(X)
	
	# cycle vars
	diff = 1
	learningRate = 1.0
	step = 0.1
	
	# determing best-fitting learningRate using brute-force	
	while diff > 0.000000000001:
		for i in range(9):
			if diff <= 0.0001:
				break
			learningRate = learningRate - step
			[history, iterations, thetasZeroStrg, thetasOneStrg] = gradientDescent(X, y, 0.5)
			diff = calcAccuracy(X, y, False)
		step = step * 0.1
	print("learningRate:{}".format(learningRate))
	return ([history, iterations, thetasZeroStrg, thetasOneStrg])

def     main(dataset):
	global thetas

	# reading data from a file, except header line
	data = np.genfromtxt(dataset[0], delimiter=',', skip_header=1)

	Y = data[:, -1]
	X = data[:, :-1]
	X_old = data[:, :-1]
	
	thetas = np.zeros(X.shape[1] + 1)
	
	# normalizing features
	X = featureScaling(X)
	#X = meanNormalization(X)

	# Computing thetas
	[history, iterations, thetasZeroStrg, thetasOneStrg] = computeThetas(X, Y)
#	normalEquation(X, Y)

	# plotting cost function results
	plt.plot(iterations, history)
	plt.ylabel('Function cost')
	plt.xlabel('Iterations')
	plt.show()
	
	if X.shape[1] == 1:
		# plotting 2D graph, with prediction line
		plt.figure(1)	
		data, = plt.plot(X, Y, 'bo', label='Training data')
		dummy_x = np.linspace(np.min(X), np.max(X), 100)
		plt.plot(dummy_x, thetas[0] + (dummy_x * thetas[1]), 'r')
		plt.ylabel('Price')
		plt.xlabel('Mileage')	
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
	# we can define what type of normalization do we apply: mean or feature, by introducing some kind of a flag
    if (len(sys.argv) > 1):
        main(sys.argv[1:])
    else:
        print("No data is passed")
        exit(-1)
