import csv

class Router(object):
	def  __init__(file):
		"""
		This will load in / process the file
		"""
		self.data = {} # make it a dict
		with open(file) as fin:
			fin = csv.reader(file, delimiter='/')

			for x in fin:
				print(x)

