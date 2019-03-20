"""
An implementation of the ID3 (decision tree) algorithm in the language of Python3.

@author Ron Rounsifer
"""
import math # for log
from collections import Counter # for first case
import Node # node structure
import time # to time execution

def ID3(attributes, data, target, count):
	"""
	The main entry point of the algorithm that will be called recrusively to create the tree.
    
	Returns the final tree structure
	"""
	count += 1 # for debugging purposes
	data = data[:]
	target_index = int(attributes.index(target))
	values = [obs[target_index] for obs in data ]
	# Check both base cases
	# case one - all obs have same class
	# return - that class
	#
	# case two - there are not attributes to test
	# return - the most common class

	if not data or (len(attributes) - 1) <= 0:
		return case_one(data)
	elif values.count(values[0]) == len(values):
		return values[0]
	else:
		# determine the best attribute by information gain
		best = best_attribute(attributes, data, target)

        # create the tree with the best attribute at the root
		decision_tree = {best:{}}
		# for each value in the attribute field, create a new node
		for value in get_values(data, attributes, best):

			# create a tree for current value
			examples = get_examples(data, attributes, best, value)
			new_attributes = attributes[:]
			new_attributes.remove(best)
			s_tree = ID3(new_attributes, examples, target, count)
            
			decision_tree[best][value] = s_tree
			
		return decision_tree
def case_one(data):
    """
    Check if all of the observations in the data have the same class.
    
    Returns
        class - str - the only classification
        OR
        Nothing
    """
    only_class = None
    class_count = Counter()
    for classification in data:
        class_count[classification[-1]] += 1
    
    if len(class_count) == 1:
        only_class = list(class_count)[0]
        
    return only_class
def best_attribute(attributes, data, target):
    """
    Determines the best attribute to split the tree at. This is determined
    by the information gain that each attribute contributes to the entropy
    of the system.
    
    Returns
        best - str - attribute with most information gain
    """
    best = attributes[0] # placeholder
    max_gain = 0.0
    attr_gain = 0.0
    for a in attributes[:]:
        attr_gain = gain(attributes, data, a, target)
        if attr_gain > max_gain and a != 'classification':
            max_gain = attr_gain
            best = a
        
    return best
def gain(attributes, data, a, target):
    """
    Calcuates the gain of each attribute passed, negating the target
    attribute. 

    Returns:
    	info_gain - float - the total info gain for the target attribute
    """
    value_counts = {}
    value_entropy = 0.0
    
    # index of the attribute
    i = attributes.index(a)
    
    # get frequency of each value in the target attribute
    for obs in data:
        if (obs[i] in value_counts.keys()):
            value_counts[obs[i]] += 1.0
        else:
            value_counts[obs[i]] = 1.0
    
    
    # Now calculate the entropy of subsetted data for each unique value
    # present in the target attribute
    for value in value_counts.keys():
        prob = float(value_counts[value]) / len(data)
        
        subsetted_data = [obs for obs in data if obs[i] == value]
        value_entropy += prob * entropy(attributes, subsetted_data, target)
        
    return (entropy(attributes, data, target) - value_entropy)
def entropy(attributes, data, target):
    """
    Calculates the entropy of the data passed for the target attribute.

    Returns:
    	value_entropy - float - the gain the value provides to the attribute
    """
    value_counts = {}
    value_entropy = 0.0
    
    # index of the attribute
    i = attributes.index(target)
    
    # get frequency of each value in the target attribute
    for obs in data:
        if (obs[i] in value_counts.keys()):
            value_counts[obs[i]] += 1.0
        else:
            value_counts[obs[i]] = 1.0
            
    # calculate entropy
    for counts in value_counts.values():
        prob = counts / len(data)
        value_entropy += (-prob) * math.log(prob, 2)

    return value_entropy
def get_values(data, attributes, best):
    """
    Get a list of values found in the best attribute column
    of the data.
    
    Return
        List of values found in the best attribute
    """
    values = []
    i = attributes.index(best)
    for obs in data:
        if obs[i] not in values:
            values.append(obs[i])
    return values
def get_examples(data, attributes, best, val):
	"""
	Get the list of all values in the best column that num_matches the value.

	Returns
		examples - list - list of all values in the best column
	"""
	examples = [[]]
	num_matches = []
	i = attributes.index(best)

	# find value num_matcheses
	for obs in data:
		if obs[i] == val:
			num_matches = []
			# add all values besides the best column
			for j in range(len(obs)):
				if j == i:
					pass
				else:
					num_matches.append(obs[j])
			examples.append(num_matches)
	examples.remove([])

	return examples
def test_model(data, attributes, decision_tree, **kwargs):
	"""Test decision tree

	Test the decision tree passed as an argument to see how well it performs
	on previously unseen data.
	"""
	count = 0
	result = ""

	# track num of obs we cannot process
	cannot_process = 0

	# track num of correct predictions
	num_matches = 0

	# calculate frequencies of each class value
	frequencies = {}
	for obs in data:
		if obs[-1] in frequencies.keys():
			frequencies[obs[-1]] += 1.0
		else:
			frequencies[obs[-1]] = 1.0
	
	# confusion matrix like below to analyze predictions
	#
	#				poor  acceptable   good   vgood
	#	poor		 0			0		 0		0
	# acceptable	 0			0		 0		0
	#	good		 0			0		 0		0
	#	vgood		 0			0		 0		0
	#
	confusion_matrix = {clas : { c : 0.0 for c in list(frequencies.keys()) } for clas in list(frequencies.keys())}

	# for each observation
	for obs in data:
		
		# work down a copy of the tree to classify the data
		temp_tree = decision_tree.copy()
		while (isinstance(temp_tree, dict)):
			
			# create current node
			value = list(temp_tree.keys())[0]
			children = temp_tree[list(temp_tree.keys())[0]]
			root = Node.Node(value, children)
			
			# narrow tree to correct sub-trees
			temp_tree = temp_tree[root.value]

			#index/value of the attribute to look at
			i = attributes.index(root.value)
			value = obs[i]

			# a sub-branch has been found at the next node
			if value in list(temp_tree.keys()):
				child = Node.Node(value, temp_tree[value])
				result = temp_tree[value]
				temp_tree = temp_tree[value]
			else:
				# cannot find a sub-branch or result
				cannot_process += 1
				break;

		if result == obs[-1]:
			num_matches += 1


		# track stats if wanted by user
		if 'statistics' in list(kwargs.keys()):
			actual = data[count][-1]
			predicted = '%s' % result
	
			try:
				confusion_matrix[actual][predicted] += 1.0
			except KeyError:
				pass

		count += 1

	# only display stats if user wants them
	if 'statistics' in list(kwargs.keys()):
		n = len(data) - cannot_process
		statistics(confusion_matrix, num_matches, n, cannot_process)
				
def statistics(confusion_matrix, num_matches, N, cannot_process):
	""" Simple statistics

	Print out simple stats such as:
		- accuracy
		- precision
		- recall
	"""
	print('')
	print('Data analysis')
	print('==============')
	print('training data: 	train.txt 	(1237 x 7)')
	print('testing data: 	test.txt 	(493 x 7)')
	print()
	print('There were %s observations that were unable to be processed correctly. These have been removed.' % (cannot_process))

	accuracy = 0.0
	precision = 0.0
	recall = 0.0
	temp_list = []
	row_sum = 0.0
	col_sum = 0.0

	print()
	print()
	print('Simple Statistics')
	print()
	# calcuate accuracy
	accuracy = (num_matches/N)
	print(f'Accuracy:	{accuracy:.2g} ( {num_matches} / {N} )')

	# calculate precision
	for key in list(confusion_matrix.keys()):
		row_sum = sum(confusion_matrix[key].values())
		precision = confusion_matrix[key][key] / row_sum
		temp_list.append(precision)
		average_precision = sum(temp_list)/len(temp_list)
	print(f'Precision:	{average_precision:.2g}')

	# calculate recall
	temp_list = []
	for i in range(len(list(confusion_matrix.keys()))):
		col_sum = 0
		for row in list(confusion_matrix.keys()):
			column = list(confusion_matrix.keys())[i]
			col_sum += confusion_matrix[row][column]
		recall = confusion_matrix[column][column] / col_sum
		temp_list.append(recall)
		average_recall = (sum(temp_list)/len(temp_list))
	print(f'Recall:		{average_recall:.2g}')
def read_data(file):
    """
    Reads the data in and returns two lists: the attributes and the data
    
    returns:
        attributes - list of the attributes
        data - list of the data
    """
    attributes = []
    data = []

    with open(file, 'r') as f:
        attributes = f.readline().split(',')
        data = f.readlines()
        
    attributes = [attr.strip() for attr in attributes]
    data = [obs.rsplit()[0].split(',') for obs in data]
    
    return attributes, data
def execute(train, test):
	"""
	Execute the ID3 algorithm on the training data.
	Test the tree on the testing data.
	Created for the sole purpose of timing the execution.
	"""

	attr_train, data_train = read_data(train)
	attr_test, data_test = read_data(test)

	# run the algorithm by passing the attributes and data
	tree = ID3(attr_train, data_train, 'classification', 0)
	#show_tree(tree)
	test_model(data_test, attr_test, tree, statistics=True)


def show_tree(decision_tree):
	for key in decision_tree.keys():
		print( key , list(decision_tree[key]))
		for i in decision_tree[key]:
			print(i, decision_tree[key][i])
			print()
		
		
begin = time.process_time()
execute('./data/train.txt', './data/test.txt')
end = time.process_time()
exec_time = (end-begin)
print(f'Time: 		{exec_time:.5g}s on CPU\n')