import numpy as np # for linear algebra
import time # to time the training period

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
		self.theta_one = np.random.rand(50,64)	
		self.theta_two = np.random.rand(50,10)
		
		# how long it took to train the model
		self.time_to_train = None

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
	
	def gradient_descent(self, theta_one, theta_two, alpha, x, y):
		num_obs = x.shape[0]
	
		# Apply weights to input layer
		hidden_layer = self.sigmoid(np.dot(theta_one, x.T))
		print(hidden_layer[:1])	
		# Apply weights to hidden layer		
		output_layer = self.sigmoid(np.dot(hidden_layer.T, theta_two))
		
		# Update layer 2 weights based on outputs
		grad = np.matmul(y.T, (output_layer - y) ) / num_obs
		theta_one = theta_one - (alpha * sum(sum(grad)))
		theta_two = theta_two - (alpha * sum(sum(grad)))		
		
		
#		print(f"one: {theta_one[:1]}")
#		print(f"two: {theta_two[:1]}")
		return theta_one, theta_two
		
	def new_cost(self, x, y, theta):
		m = x.shape[0]
		h = self.sigmoid(np.matmul(x, theta))
		cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1-y.T), np.log(1-h))) / m
		return cost
		
	def learn(self, iterations, learning_rate, fout):
		"""
		Train the network on the data
		"""
		begin = time.process_time()	
		self.init_bias()

		X = self.training_data
		Y = self.y_vectorized
		print("Building model...")
		
		
		# Since I am using linear algebra, all of the errors are propogated
		# back into the weights immediately (backprop)
		for i in range(iterations):
			self.theta_one, self.theta_two = self.gradient_descent(self.theta_one, self.theta_two, learning_rate, X, Y)
		end = time.process_time()
		self.time_to_train = end-begin
		print("Model complete.")
#		np.savetxt(f"./models/{fout}", self.theta, delimiter=',')
	
	def log(self, training, testing, iterations, alpha, correct, num_preds):
		accuracy = correct / num_preds
		with open("./logs/tests.txt", 'a') as f:
			f.write("============================================\n")
			f.write(f"Training data: {training}\n")
			f.write(f"Testing data: {testing}\n")
			f.write(f"Num iterations: {iterations}\n")
			f.write(f"Learning rate: {alpha}\n")
			f.write(f"Accuracy: {accuracy}\n")
			f.write(f"Model building time: {self.time_to_train}\n")
	
	def test(self, training, testing, iterations, alpha):
		predictions = []
		hidden_layer = self.sigmoid(np.dot(self.theta_one, self.testing_data.T))
		output_layer = self.sigmoid(np.dot(hidden_layer.T, self.theta_two))

		for prediction in output_layer:
			predictions.append(np.argmax(prediction))
		
		# count num correct
		correct = 0
		num_preds = len(predictions)
		for index in range(num_preds):
			if predictions[index] == self.testing_y[index]:
				correct += 1
		self.log(training, testing, iterations, alpha, correct, num_preds)


train = "./data/training.txt"
test = "./data/testing.txt"
fout = "handwriting.nn"
iterations = 1000
learning_step = 0.05
net = Neural_Network()
net.load_data(train, test)
net.learn(iterations, learning_step, fout)
net.test(train, test, iterations, learning_step)
