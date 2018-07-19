import numpy as np
import sys
import matplotlib.pyplot as plt

thetas = np.zeros(2)

def 	featureScaling(data):
	max_value = np.amax(data)
	return (np.divide(data, max_value))

def 	meanNormalization(data):
	return (np.divide((data - np.mean(data)), np.amax(data)))

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
	thetasHistory = list()
	iterations = list()

	for i in range(iterationsNum):
		J = (h_function(X) - Y).dot(X)
		J = learningRate * np.divide(J, m)
		thetas = thetas - J
		thetasHistory.append(computeCost(X, Y))
		iterations.append(i)

	return (iterations, thetasHistory)

def	addBiasUnit(arr):
	bias_arr = np.ones((arr.shape[0],), dtype=float)
	return (np.vstack((bias_arr, arr)).T)

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
	
	diff = 1
	learningRate = 1.0
	step = 0.1
	
	# determing best-fitting learningRate using brute-force	
	while diff > 0.0001:
		for i in range(9):
			if diff <= 0.0001:
				break
			learningRate = learningRate - step
			[history, iterations] = gradientDescent(X, y, 0.5)
			diff = calcAccuracy(X, y, False)
		step = step * 0.1
	return ([history, iterations])

def     main(dataset):
	global thetas

	# reading data from a file
	data = np.genfromtxt(dataset[0], delimiter=',')
	# extracting first row with columns explanation
	data = np.delete(data, (0), axis=0)

	Y = data[:, 1]
	X = data[:, 0]
	
	# normalizing features
	X = featureScaling(X)
	#X = meanNormalization(X)

	[history, iterations] = computeThetas(X, Y)
#	normalEquation(X, Y)

	plt.plot(iterations, history)
	plt.ylabel('Function cost')
	plt.xlabel('Iterations')
	plt.show()
	
	data, = plt.plot(X, Y, 'bo', label='Training data')
	dummy_x = np.linspace(np.min(X), np.max(X), 100)
	plt.plot(dummy_x, thetas[0] + (dummy_x * thetas[1]), 'r')
	plt.ylabel('Price')
	plt.xlabel('Mileage')
	plt.legend()
	plt.show()
	# saving thetas to temp file
	with open('thetas.txt', 'w') as f:
		for i in range(thetas.shape[0]):
			f.write(str(thetas[i]) + "\n")

if __name__ == '__main__':
	# we can define what type of normalization do we apply: mean or feature, by introducing some kind of a flag
    if (len(sys.argv) > 1):
        main(sys.argv[1:])
    else:
        print("No data is passed")
        exit(-1)
