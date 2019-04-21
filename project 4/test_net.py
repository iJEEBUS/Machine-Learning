import numpy as np

class Neural_Net():

	def __init__(self):
		"""Constructor
		
		Inits the variables needed to be used during the building/using of the neural
		network.
		"""
		self.INPUT_LAYER_SIZE = 63
		self.HIDDEN_LAYER_SIZE = 50
		self.OUTPUT_LAYER_SIZE = 10

		self.NUM_ITERATIONS = 500	
		self.LEARNING_RATE = 0.05		


		self.hidden_layer = None
		self.output_layer = None

		# Init the weights
		self.theta_one = np.random.randn(self.INPUT_LAYER_SIZE, self.HIDDEN_LAYER_SIZE) * \
					np.sqrt(2.0/self.INPUT_LAYER_SIZE)
		self.theta_two = np.random.randn(self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_SIZE) * \
					np.sqrt(2.0/self.HIDDEN_LAYER_SIZE)
		# Init the biases
		self.hidden_bias = np.full((1, self.HIDDEN_LAYER_SIZE), 0.1)
		self.output_bias = np.full((1, self.OUTPUT_LAYER_SIZE), 0.1)

		
	def relu(self, matrix):
		"""ReLU

		The activation function for each layer. This will re-write the contents of the matrix.
		If the number is less than 0, set that number to 0. Otherwise, keep the positive number.

		Returns:
			matrix (ndarray) - relu applied matrix
		"""
		return np.maximum(0, matrix)

	def relu_prime(self, matrix):
		"""ReLU derivative

		Returns gradient of the inputtted matrix.
		All negatives are set to 0.
		All positives are set to 1.
		"""
		matrix[matrix < 0] = 0
		matrix[matrix > 0] = 1
		return matrix


	def cost(self, pred, actual):
		"""Prediction error

		Calculates the cost of the current network. The cost is how far off the network is from
		the correct answer. This will allow me to change the weight layers.
		"""
		cost = np.sum((pred - actual)**2) / 2.0
		return cost


	def cost_prime(self, pred, actual):
		"""Prediction error derivative

		The derivative of the cost (i.e. the slope of the cost)
		"""
		return pred - actual


	def feed_forward(self):
		"""Feeding forward data

		Feed the data forward through the network.
		1 - apply first layer of weights to input
			1a - apply ReLU
		2 - apply second layer of weights to the hidden layer
			2a - apply ReLU

		Returns
			matrix (ndarray) - the networks guesses after feeding forward.
		"""
		self.hidden_layer = np.dot(self.training_data, self.theta_one) + self.hidden_bias
		hidden_layer_relu = self.relu(self.hidden_layer)
		
		self.output_layer = np.dot(hidden_layer_relu, self.theta_two) + self.output_bias
		outputs = self.relu(self.output_layer)
		return outputs


	def backprop(self):
		"""Propagate error backwards

		After feeding forward, this will calculate the error of the network and go about
		propogating that back to the layer weights (this is possible thanks to the chain rule)
		"""

		yHat = self.feed_forward()
	
		# Layer errors
		error_output = (yHat - self.y_vectorized) * self.relu_prime(self.output_layer)
		error_hidden = np.dot(error_output, self.theta_two.T) * self.relu_prime(self.hidden_layer)
	
		# Cost derivative for weights
		deriv_output = np.dot(self.hidden_layer.T, error_output) 
		deriv_hidden = np.dot(self.training_data.T, error_hidden)

#		deriv_hidden = np.dot(error_hidden, self.training_data)
		# Update weights
		self.theta_one -= self.LEARNING_RATE * deriv_hidden
		self.theta_two -= self.LEARNING_RATE * deriv_output

	def vectorize(self, data):
		"""
		Helper method that converts inputted numbers into vectorized versions
		(e.g. 7 = [0,0,0,0,0,0,0,1,0,0])
		"""
		final_list = []
		temp = [0]*10

		for y in data:
			for index in range(len(temp)):
				if int(y) == index:
					temp[index] = 1

		final_list.append(temp)
		temp = [0] * 10
		return np.array(final_list)	


	def normalize(self, matrix):
		"""
		Helper method to normalize the data in the supplied matrix.
		This makes use of Min-Max Normalization.
		"""
		temp = []
		for obs in matrix:
 			n = (obs - obs.min()) / (np.ptp(obs))
 			temp.append(n)
		return np.array(temp)


	def learn(self):
		"""
		Main entry point of the program. 
		Feed forward and back prop the error for the specified number of iterations.
		"""
		print("Learning...")
		for i in range(self.NUM_ITERATIONS):
			self.backprop()
		
		print("Model complete.")
		self.test()
		print("Testing...")

	def test(self):
		
		# load data
		hidden_layer = np.dot(self.testing_data, self.theta_one)
		outputs = np.dot(hidden_layer, self.theta_two)
	
		counter = 0
		correct = 0
		for pred in list(outputs):
			guess = list(pred).index(max(list(pred)))
			
			if guess == self.testing_y[counter]:
				correct += 1
			counter += 1
		print(correct/self.testing_data.shape[0])
			
	def load_data(self, training, testing):
		"""
		Helper method to load in the data
		"""	
		# Create ndarrays from the data
		training_data = np.genfromtxt(training, delimiter=' ')
		testing_data = np.genfromtxt(testing, delimiter=' ')
	
		# The index of the class col
		classification = training_data.shape[1] - 1
	
		# Get all of the classification cols
		self.training_y = training_data[:, classification]
		self.testing_y = testing_data[:, classification]
	
		# Get only the data, no classes
		self.training_data = self.normalize(training_data[:,:classification-1])
		self.testing_data = self.normalize(testing_data[:,:classification-1])
	
		# Vectorize the training solutions
		self.y_vectorized = self.vectorize(self.training_y)

		# Setting private variables
		self.INPUT_LAYER_SIZE = self.training_y.shape[0]	
nn = Neural_Net()
nn.load_data("./data/training.txt", "./data/testing.txt")
nn.learn()
