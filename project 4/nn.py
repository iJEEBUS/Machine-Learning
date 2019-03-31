import math

class Neural_Net():
	
	def __init__(self):
		self.y = []
		self.y_vector = []
		self.X = []
		self.weights_one = []
		self.weights_two = []	
		self.data = []


	def read_data(self, data):
		
		features = 0
		classes = 10

		with open(data) as f:
			for line in f:
				line = line.split(' ')
				y = line[-1]
				data = line[:len(line)-1]
				if len(data) != 0:
					self.y.append(int(y))
					self.data.append(data)
					
					# vectorize the outputs (3823 x 10)
					vector = [0] * 10 # because we are only looking at 10 digits 
					vector[int(y)] = 1
					self.y_vector.append(vector)
			
					data = [1.0] + data[:]						
					for i in range(1, len(data)):
						data[i] = int(data[i])
					self.X.append(data[:])
	
		features = len(self.X[0])-1
		print("	This data contains:")
		print(f"	Features: {features}")
		print(f"	Classifications: {classes}")

	def init_weights(self):
		num_nodes = len(self.X[0])-1
		self.weights_one = [[1.0]*(num_nodes+1)]*num_nodes
		self.weights_two = [[1.0]*num_nodes]*10
		print(len(self.weights_two), len(self.weights_two[0]))
	def matrix_mul(self,a,b):
		"""
		Multiply two matrices together
		
		Returns
			The product matrix
		"""
		# Check dimensions
		if len(a[0]) != len(b):
			print("Matrices are not of an appropriate size to multiply")
			print(len(a), len(a[0]))
			print(len(b), len(b[0]))
			return
		
		dot_products = []	
		for row in range(len(a)):
			for col in range(len(b[row])):
				v_one = a[row]
				v_two = self.get_column(b, col)
				value = self.dot_product(v_one, v_two)
				dot_products.append(value)	
		
		# Create the final matrix from the vector of dot products	
		new_matrix = [ [0]*len(a) ] * len(b[0])
		new_matrix = [ dot_products[x:x+len(b[0])] for x in range(0, len(dot_products), len(b[0])) ]	
		
		return new_matrix

	def dot_product(self, a,b):
		
		dot_product = sum([float(x) * float(y) for x, y  in zip(a,b) ])
		return dot_product 
	
	def get_column(self, matrix, target):
		
		column = []
		
		for row in range(len(matrix)):
			column.append(matrix[row][target])
		return column

	def sigmoid(self, matrix):
		"""
		Element-wise sigmoid function on a matrix
		"""
		new_matrix = [[0]*len(matrix[0])]*len(matrix)
		for row in range(0, len(matrix)):
			for col in range(0, len(matrix[0])):
				new_matrix[row][col] = 1 / (1 + math.exp(-matrix[row][col]) )
		
		return new_matrix
	
	def transpose(self, matrix):
		"""
		Transpose the matrix passed as an argument
		
		Returns
			A transposed matrix
		"""
		return [list(x) for x in zip(*matrix)]

	def learn(self, n, data):
		# one = [[1,2,3],[4,5,6]]
		# two = [[7,8], [9, 10], [11,12]]	
		# x_one = self.matrix_mul( one, two )
#		x_one = self.matrix_mul(self.weights_one, self.transpose(self.X))
		print("Loading data...")
		self.read_data(data)
		print("Initializing weights...")
		self.init_weights()	
		x_one = self.sigmoid(self.matrix_mul(self.sigmoid(self.weights_one), self.transpose(self.sigmoid(self.X))))
		biases = [1.0] * len(x_one[0])
		x_one = biases[:] + x_one[:]
		x_two = self.sigmoid(self.matrix_mul(self.weights_two, x_one))
		print(len(x_one), len(x_one[0]))
		print(len(x_two), len(x_two[0]))
#		print(len(self.X), len(self.X[0]))
n = Neural_Net()
n.learn(10000, "./data/training-half.txt")
