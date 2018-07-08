import numpy as np
from train import h_function

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
			X = np.vstack((np.ones((1,), dtype=float), X)).T
			result = h_function(X)
			print(X)
			print(result)
			print("Predicted cost for a car with mileage({}): {}".format(''.join(str(el) for el in X[:, 1]), ''.join(str(el) for el in result)))
			return 


if __name__ == '__main__':
	main()