import numpy as np
from train import addBiasUnit
import sys
import argparse

def	featureScaling(X, amax):
	X = np.array([X / amax])
	return (X)

def	meanNormalization(X, avg, stddev):
	X = np.array([np.divide((X - avg), stddev)])
	return (X)

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
				if args.is_fscale:
					features.append([featureScaling(float(row[i]), amax[i]) for i in range(len(row))])
				else:
					features.append([meanNormalization(float(row[i]), avg[i], stddev[i]) for i in range(len(row))])
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

def 	main(data):
	thetas = getThetasFromFile()
	amax, avg, stddev = getMetricsFromFile()
	if data is None and len(thetas) is 2:
		while 1:
			try:
				X = float(input('Mileage for car: '))
			except ValueError:
				print("Not a number")
			else:
				if args.is_fscale:
					upX = np.array([featureScaling(X, amax[0])])
				else:
					upX = np.array([meanNormalization(X, avg[0], stddev[0])])
				X = np.ndarray((1,1), buffer=np.array(X))
				upX = np.ndarray((1,1), buffer=np.array(upX))
				calcResult(upX, thetas, X)
				return
	elif data is not None:
		upX, X = getDataFromFile(data, amax, avg, stddev)
		calcResult(upX, thetas, X)
	else:
		print("No possibility to pass multiple features via console")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict values considering retrieved thetas.')
	parser.add_argument('-meanNorm', dest='is_fscale', action='store_false', default=True, help='choose mean normalization as method to rescale input data')
	parser.add_argument('-fscale', dest='is_fscale', action='store_true', default=True, help=' [default] choose feature scalling as method to rescale input data')
	parser.add_argument('-data', dest='dataset', default=None, help='dataset with input values to predict output')
	args = parser.parse_args()
	main(args.dataset)
	#else:
	#	print("Usage: python " + __file__ + " [data.csv]")
	#	print("\tdata.csv(optional) -- is a file with data to predict. Separate passed data with comma")
	#	print("\tIf file isn't specified data is expected to be entered via console")
	#	main()
