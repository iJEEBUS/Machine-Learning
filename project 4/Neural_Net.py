"""
A single-layer neural network using linear algebra that is able to classify 
handwritten digits.

Accuracy: ~92%
Time to build: < ~1.5 seconds

@author Ron Rounsifer
"""
import numpy as np # for linear algebra
import time # to time the training period
import pickle # used for storing model in a file

class Neural_Network():
	
	def __init__(self, training, testing, iterations, alpha):
		self.model_loaded = False
		self.TRAINING = training
		self.TESTING = testing
		self.EPOCHS = iterations
		self.LEARNING_RATE = alpha
		# Create ndarrays from the data
		training_data = np.genfromtxt(training, delimiter=' ')
		testing_data = np.genfromtxt(testing, delimiter=' ')
		
		# The index of the class col
		classification = training_data.shape[1] - 1
		self.training_y = training_data[:, classification]
		self.testing_y = testing_data[:, classification]
		self.training_data = self.normalize(training_data[:,:classification-1])
		self.testing_data = self.normalize(testing_data[:,:classification-1])
		self.y_vectorized = self.vectorize(self.training_y)
		self.init_bias()

		# Weights
		self.theta_one = np.random.rand(64,10)	
		self.theta_two = np.random.rand(10,10)
		
		# how long it took to train the model
		self.time_to_train = None
		
	
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
	
		# hidden layer
		hidden_layer = self.sigmoid(np.matmul(x, theta_one))
		grad = np.matmul(x.T, (hidden_layer - y)) / num_obs
		theta_one = theta_one - (alpha * grad)

		# output layer
		output_layer = self.sigmoid(np.matmul(hidden_layer, theta_two))
		grad = np.matmul(y.T, (output_layer - y)) / num_obs
		theta_two = theta_two - (alpha * grad)
		return theta_one, theta_two
		

	def new_cost(self, x, y, theta):
		m = x.shape[0]
		h = self.sigmoid(np.matmul(x, theta))
		cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1-y.T), np.log(1-h))) / m
		return cost
		

	def learn(self, fout):
		"""
		Train the network on the data
		"""
		begin = time.process_time()	

		X = self.training_data
		Y = self.y_vectorized
		print("Building model...")
		
		for i in range(self.EPOCHS):
			self.theta_one, self.theta_two = self.gradient_descent(self.theta_one, self.theta_two, self.LEARNING_RATE, X, Y)
		end = time.process_time()
		self.time_to_train = end-begin
		print("Model complete.")
		model = [self.theta_one, self.theta_two]
		self.save_model(model, fout)
		self.model_loaded = True
	

	def save_model(self, m, fout):
		with open (f"./models/{fout}", 'wb') as f:
			pickle.dump(m, f)


	def load_model(self, fin):
		print("Loading model...")
		try:
			with open (f"./models/{fin}", 'rb') as f:
				model = pickle.load(f)
				self.theta_one = model[0]
				self.theta_two = model[1]
			self.model_loaded = True
		except Exception as e:
			print("It looks like this model has not been created yet")
			print(e)

	def log(self, training, testing, iterations, alpha, correct, num_preds):
		
		accuracy = correct / num_preds

		print("============================================\n")
		print(f"Training data: {training}\n")
		print(f"Testing data: {testing}\n")
		print(f"Num iterations: {iterations}\n")
		print(f"Learning rate: {alpha}\n")
		print(f"Accuracy: {accuracy}\n")
		print(f"Model building time: {self.time_to_train}\n")

		with open("./logs/tests.txt", 'a') as f:
			f.write("============================================\n")
			f.write(f"Training data: {training}\n")
			f.write(f"Testing data: {testing}\n")
			f.write(f"Num iterations: {iterations}\n")
			f.write(f"Learning rate: {alpha}\n")
			f.write(f"Accuracy: {accuracy}\n")
			f.write(f"Model building time: {self.time_to_train}\n")
	
	def test(self):
		if self.model_loaded:
			predictions = []
			hidden_layer = self.sigmoid(np.matmul(self.testing_data, self.theta_one))
			output_layer = self.sigmoid(np.matmul(hidden_layer, self.theta_two))

			for prediction in output_layer:
				predictions.append(np.argmax(prediction))
		
			# count num correct
			correct = 0
			num_preds = len(predictions)
			for index in range(num_preds):
				if predictions[index] == self.testing_y[index]:
					correct += 1
			self.log(self.TRAINING, self.TESTING, self.EPOCHS, self.LEARNING_RATE, correct, num_preds)



###### TESTING ######
fout = "handwriting.nn"
net = Neural_Network("./data/training.txt", "./data/testing.txt", 500, 0.07)
net.learn(fout)
net.test()
#net.load_model(fout)
#net.test()
