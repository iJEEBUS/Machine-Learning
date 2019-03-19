class Node():
    value = ''
    children = []
    
    def __init__(self, val, dictionay):
        self.set_value(val)
        self.create_children(dictionary)
    
    def __str__(self):
        return str(self.value)
    
    def set_value(self, val):
        self.value = val
    
    def create_children(self, dictionary):
        if (isinstance(dictionary, dict)):
            self.childre = dictionary.keys()