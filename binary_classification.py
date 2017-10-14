from random import gauss, randrange
from math import exp

# Neural Network
#   O Prediction
#  / \ Weights + bias
# O   O Input_Paramters

# Sigmoid Function to limit the output between 0 and 1
def sigmoid(x):
	return 1/(1 + exp(-x));

# Derivative of the sigmoid function
def d_sigmoid(x):
	return sigmoid(x)*(1 - sigmoid(x))

# Initialize the parameters to random numbers guassian distribution to let numbers be smaller
w1 = gauss(0, 1)
w2 = gauss(0, 1)
b = gauss(0, 1)

# Define the trainging set with the two parameters and the "right answer"
trainig_set = [[3, 5, 1], [4, 4, 1], [6, 3, 0], [6.5, 2, 1], [6, 0.5, 1], [4.5, 3, 1], [2, 2, 1], [5, 1, 1], [8, 3, 0], [5, 6, 0], [2.8, 7, 0], [7.5, 4, 0], [9, 3.5, 0], [10, 5, 0], [7, 6, 0]]

# Defining a learning rate to take a fraction of the descent
learning_rate = 0.2

# Pick random example from training set and try to evaluate the prediction
for _ in range(50000):
	random_example = randrange(len(trainig_set))
	example = trainig_set[random_example]

	# Calculate the hypothsis
	z = w1*example[0] + w2*example[1] + b

	# Predict the value by compressing the value of hypothesis
	prediction = sigmoid(z)

	# Obtain target
	target = example[2]

	# Squared Error
	cost = (prediction - target) ** 2

	# Derivative of Squared error with respect to prediction
	dcost_dprediction = 2 * (prediction - target)

	# Derivative of prediction with respect to hypothesis
	dprediction_dz = d_sigmoid(z)

	# Partial derivative of hypothesis with respect to argument 1, argument 2 and bias
	dz_dw1 = example[0]
	dz_dw2 = example[1]
	dz_db = 1

	# Derivative of Squared error with respect to hypothesis (Chain Rule)
	dcost_dz = dcost_dprediction*dprediction_dz

	# New value of the parameters changed to a fraction of the descent at the example point
	w1 = w1 - learning_rate*(dcost_dz*dz_dw1)
	w2 = w2 - learning_rate*(dcost_dz*dz_dw2)
	b = b - learning_rate*(dcost_dz*dz_db)

#Once the parameters are obtained the model is trained.
print(w1)
print(w2)
print(b)