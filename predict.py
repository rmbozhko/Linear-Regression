import numpy as np
from train import h_function, featureScaling, meanNormalization
def main():
	while 1:
		try:
			X = float(input('Mileage for car: '))
		except ValueError:
			print("Not a number")
		else:
			with open('thetas.txt', 'r') as f:
				thetas = f.readlines()
			thetas = [theta.strip('\n') for theta in thetas]
			thetas = [float(theta) for theta in thetas]
			X = featureScaling(X)
			X = np.vstack((np.ones((1,), dtype=float), X)).T
			result = X.dot(thetas)
			print(thetas)
			print(X)
			print(result)
			print("Predicted cost for a car with mileage({}): {}".format(''.join(str(el) for el in X[:, 1]), ''.join(str(el) for el in result)))
			return 


if __name__ == '__main__':
	main()
