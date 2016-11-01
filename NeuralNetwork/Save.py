"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import pickle

def save(net, name):
	"""
	Saves the given neural network in the file
	"""
	with open(name, 'wb') as output:
		pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)

def load(name):
	"""
	Loads a neural network from the given file
	"""
	with open(name, 'rb') as input:
		return pickle.load(input)
