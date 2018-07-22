Usage of Linear Regression Program:
Firstly, you should train thetas for prediction. This is how we do it:
python train.py dataset.csv
After following command 2 temp files will be generated:
	1. thetas.txt -- thetas values for prediction
	2. metrics.txt -- values needed for featureScaling

Then we can use retrieved thetas for value prediction, using following command:
python predict.py
There is help flag for both programs, which describes possibilities for both programs.
