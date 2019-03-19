import pandas as pd # for dataframe
from typing import NamedTuple # for struct of leaf
import math # for log

# the leaf type that will be used to build the tree
class Leaf(NamedTuple):
    attribute: str
    classification: str
    info_gain: float


class Decision_Tree(object):
    
    def __init__(self):
        self.df = None
        self.entropy = 0.0
        self.num_obs = 0
        self.num_attr = 0
        self.classes = []
        
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.num_obs = len(self.df)
        self.num_attr = len(self.df.columns) - 1
        self.classes = self.df['classification'].value_counts().keys()
        self.entropy = 0.0
        
    def load_data(self, data):
        self.df = pd.read_csv(data)
        self.num_obs = len(self.df)
        self.num_attr = len(self.df.columns) - 1
        self.classes = self.df['classification'].value_counts().keys()
        self.entropy = 0.0
        
    def case_one(self):
        
        # Case 1:
        # if all of the conditions are the same, return that value
        classes = self.df['classification'].unique()
        if len(classes) == 1:
            return Leaf(attribute=None, classification=classes[0], info_gain=None)
        return False
        
    def case_two(self):
        
        # Case 2:
        # if there are no attributes, return the most common occurrence
        if len(self.df.columns) == 1:
            return Leaf(attribute=None, classification=df['classification'].value_counts().keys()[0], info_gain=None)
        return False
        
    def case_three(self):
        
        # Case 3:
        # calculate the root
        self.calculate_entropy()
        root = self.calculate_root()
        return False
        
        
    def calculate_entropy(self):
        
        entropy = 0.0
        print(self.num_obs)
        for i in range(len(self.classes)):
            x = (self.df['classification'].value_counts()[i] / self.num_obs)
            print(self.df['classification'].value_counts()[i])
            
            # we have to use base 4 since there are 4 possible classes
            self.entropy -= (x * math.log(x, 4)) 
            print(self.entropy)
        
        
    def calculate_root(self):
        
        # the info gain of each attribute
        root = None
        max_gain = ('', 0.0)
        
        # for each attribute, calculate the entropy
        for i in range(self.num_attr):
            new_gain = self.calculate_gain(self.df.columns[i])

            if (new_gain[1] > max_gain[1]):
                max_gain = new_gain;
        
        return Leaf(attribute=max_gain[0], classification=None, info_gain=max_gain[1])
    
    def calculate_gain(self, attr):
        """
        This returns a tuple that contains the attr name and the the information gain.
        Ex:
            ("doors", 0.95)
        """
        gain = 1.0
        return (attr, gain)
        
    def run(self):
        if tree == self.case_one():
            return tree
        elif tree == self.case_two():
            return tree
        else:
            print("entered third case")
            self.case_three()