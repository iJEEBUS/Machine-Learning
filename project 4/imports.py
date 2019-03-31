import numpy as np # for linear algebra

class Neural_Network():
	
	def __init__(self):
		# The training data with solutions
		self.training_data = None
		self.training_y = None
		self.y_vectorized = None # (3000 x 10)		
	
		# Testing data with solutions (3000 x 64)
		self.testing_data = None
		self.testing_y = None
		
		# Weights
		self.theta = np.ones((64,10), dtype=float)	
		
		# Output matrix
		self.output = None
		
		# Cost of function
		self.cost = None

	def load_data(self, training, testing):
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
	
	def vectorize(self, data):
		final_list = []
		temp = [0]*10
		for y in data:
			for index in range(len(temp)):
				if int(y) == index:
					temp[index] = 1
			final_list.append(temp)
			temp = [0] * 10
	
		return np.array(final_list)
	
	def init_bias(self):
		temp_list = []
		bias = np.array([[1.0]])
		for obs in self.training_data:
			obs = np.concatenate((bias, obs), axis=None)
			temp_list.append(obs.tolist())	
		self.training_data = np.array(temp_list)
		
		temp_list = []
		for obs in self.testing_data:
			obs = np.concatenate((bias, obs), axis=None)
			temp_list.append(obs.tolist())
		self.testing_data = np.array(temp_list)

	def normalize(self, matrix):
		temp = []
		for obs in matrix:
			n = (obs - obs.min()) / (np.ptp(obs))
			temp.append(n)
		return np.array(temp)
	
	def sigmoid(self, matrix):
		return 1.0 / (1 + np.exp(-matrix))
	
	def gradient_descent(self, theta, alpha, x, y):
		m = x.shape[0]
		h = self.sigmoid(np.matmul(x, theta))
		grad = np.matmul(x.T, (h - y)) / m
		theta = theta - (alpha * grad)
		return theta
		
	def new_cost(self, x, y, theta):
		m = x.shape[0]
		h = self.sigmoid(np.matmul(x, theta))
		cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1-y.T), np.log(1-h))) / m
		return cost
		
	def learn(self, iterations, learning_rate):
		"""
		Train the network on the data
		"""
			
		print("Adding bias to input data...")
		self.init_bias()

		X = self.training_data
		Y = self.y_vectorized
		for i in range(iterations):
			self.theta = self.gradient_descent(self.theta, learning_rate, X, Y)
		
	def test(self):
		predictions = []
		model = self.sigmoid(np.matmul(self.testing_data, self.theta))
		for prediction in model:
			predictions.append(np.argmax(prediction))
		
		# count num correct
		correct = 0
		num_preds = len(predictions)
		for index in range(num_preds):
			if predictions[index] == self.testing_y[index]:
				correct += 1
		
		print(correct, (correct/num_preds))



iterations = 50000
learning_step = 0.04
net = Neural_Network()
net.load_data("./data/training.txt", "./data/testing.txt")
net.learn(iterations, learning_step)
net.test()
