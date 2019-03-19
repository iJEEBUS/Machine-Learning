import math
from collections import Counter
import Node

def ID3(attributes, data, target, count):
    """
    The main entry point of the algorithm that will be called recrusively to create the tree.
    
    Returns the final tree structure
    """
    count += 1
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
    Get the list of all values in the best column that match the value.
    """
    examples = [[]]
    match = []
    i = attributes.index(best)
    
    # find value matches
    for obs in data:
        if obs[i] == val:
            match = []
        # add all values besides the best column
        for j in range(len(obs)):
            if j != i:
                match.append(obs[j])
        examples.append(match)
    examples.remove([])
    return examples

def test_model(data, attributes, decision_tree):
	count = 0
	result = ""
	match = 0

	for obs in data:
		count += 1
		temp_tree = tree.copy()

		while (isinstance(temp_tree, dict)):
			#print(temp_tree.keys()[0])
			root = Node.Node(list(temp_tree.keys())[0], temp_tree[list(temp_tree.keys())[0]])
			temp_tree = temp_tree[list(temp_tree.keys())[0]]
			i = attributes.index(root.value)
			value = obs[i]

			if value in list(temp_tree.keys()):
				child = Node.Node(value, temp_tree[value])
				result = temp_tree[value]
				temp_tree = temp_tree[value]
			else:
				print("Cannot process input %s" % (count) )
				break;
		if result == obs[-1]:
			print("%s) %s = %s <------ MATCH" % (count, obs[-1], result))
			match += 1
		else:
			print("%s) %s = %s" % (count, obs[-1], result))
	print("Accuracy: %s" % (match/len(data)))
		


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

attr_train, data_train = read_data('./data/train.txt')
attr_test, data_test = read_data('./data/test.txt')

# run the algorithm by passing the attributes and data
tree = ID3(attr_train, data_train, 'classification', 0)
test_model(data_test, attr_test, tree)