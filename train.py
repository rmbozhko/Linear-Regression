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
	print("Cost function result: {}".format(J))
	return (J)

def 	h_function(X):
	global thetas
	return (X.dot(thetas))

def 	trainThetas(X, Y, learningRate=0.001, iterationsNum=1500):
	global thetas
	# in case, when we have only one feature in X, we can assign m to X.size,
	# otherwise we should specify the axis of X which we are going to assign
	m = Y.shape[0]
	thetasHistory = list()

	for i in range(iterationsNum):
		J = (h_function(X) - Y).dot(X)
		J = np.divide(np.multiply(J, learningRate), m)
		thetas = thetas - J
		thetasHistory.append(computeCost(X, Y))
	return (thetasHistory)

def     main(dataset):
	global thetas

	iterationsNum = 1500
	learningRate = 0.001

	# reading data from a file
	data = np.genfromtxt(dataset[0], delimiter=',')
	# extracting first row with columns explanation
	data = np.delete(data, (0), axis=0)

	Y = data[:, 1]
	X = data[:, 0]
	#Y = featureScaling(Y)
	#X = featureScaling(X)

	Y = meanNormalization(Y)
	X = meanNormalization(X)

	# adding bias column to X data
	bias_arr = np.ones((Y.shape[0],), dtype=int)
	X = np.vstack((bias_arr, X)).T

	# visualizing data
	#Y_data, = plt.plot(Y, 'ro', label='Y values')
	#X_data, = plt.plot(X, 'bo', label='X values')
	
	#plt.legend()
	#plt.show()
	computeCost(X, Y)

	history = trainThetas(X, Y)

	plt.plot([i for i in range(iterationsNum)], history)
	plt.show()
	#plt.x_label('Cost of a function')
	#plt.y_label('Iterations')

	computeCost(X, Y)
	
	Y_data, = plt.plot(Y, 'ro', label='Y values')
	X_data, = plt.plot(X[:, 1], 'bo', label='X values')
	print(thetas)
	plt.plot(thetas)
	plt.legend()
	plt.show()
	f = open('thetas.txt', 'w')
	for i in range(thetas.shape[0]):
		f.write(str(thetas[i]) + "\n")
	f.close()


if __name__ == '__main__':
	# we can define what type of normalization do we apply: mean or feature, by introducing some kind of a flag
    if (len(sys.argv) > 1):
        main(sys.argv[1:])
    else:
        print("No data is passed")
        exit(-1)
