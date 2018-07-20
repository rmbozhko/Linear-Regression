import numpy as np
from train import addBiasUnit
import sys

def	featureScaling(X, amax):
	X = np.array([X / amax])
	return (X)

def	meanNormalization(X, avg, stddev):
	X = np.array([ (X - avg) / stddev ])
	return (x)

def	calcResult(X, thetas, X_old):
	result = list()
	
	for i in range(X.shape[0]):
		try:
			result.append(h_function(X[i].T, thetas))
		except ValueError:
			print("Error: Trained thetas don't correspond to passed data")
		else:
			print("Predicted output with following features ({}): {}".format(', '.join(str(el) for el in X_old[i, :]), ''.join(str(el) for el in result[i])))

def	getDataFromFile(filename, amax, avg, stddev):
	features = list()
	X = list()
	
	with open(filename, 'r') as f:
		for row in f:
			row = row.split(',')
			row[-1] = row[-1].strip('\n')
			try:
				X.append([float(row[i]) for i in range(len(row))])
				features.append([featureScaling(float(row[i]), amax[i]) for i in range(len(row))])
			except ValueError:
				print("Not a number was found in a file " + filename)
				exit()
	
	return (np.array(features), np.array(X))

def	getThetasFromFile(filename='thetas.txt'):
	with open(filename, 'r') as f:
		thetas = f.readlines()
	thetas = [theta.strip('\n') for theta in thetas]
	thetas = [float(theta) for theta in thetas]
	return (thetas)

def	h_function(X, thetas):
	return (addBiasUnit(X).dot(thetas))

def	getMetricsFromFile(filename='metrics.txt'):	
	# metrics vars
	amax = list()
	avg = list()
	stddev = list()

	with open(filename, 'r') as f:
		for metrics in f:
			metrics = metrics.split(' ')
			metrics[-1] = metrics[-1].strip('\n')
			if len(metrics) is not 3:
				print("Unsufficient number of metrics values")
				exit()
			amax.append(float(metrics[0]))
			avg.append(float(metrics[1]))
			stddev.append(float(metrics[2]))
	return (amax, avg, stddev)

def 	main(data=None):
	thetas = getThetasFromFile()
	amax, avg, stddev = getMetricsFromFile()
	if data is None:
		while 1:
			try:
				X = float(input('Mileage for car: '))
			except ValueError:
				print("Not a number")
			else:
				X = np.array([featureScaling(X, amax[0])])
				#X = meanNormalization(X, avg[0], stddev[0])
				calcResult(X, thetas)
				return
	else:
		upX, X = getDataFromFile(data, amax, avg, stddev)
		calcResult(upX, thetas, X)


if __name__ == '__main__':
	if (len(sys.argv) > 1):
		main(sys.argv[1])
	else:
		print("Usage: python " + __file__ + " [data.csv]")
		print("\tdata.csv(optional) -- is a file with data to predict. Separate passed data with comma")
		print("\tIf file isn't specified data is expected to be entered via console")
		main()
