"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import sys
import numpy as np
import Network
import Extraction

def test(network, images, targets):
	res = 0
	for i in range(len(images)):
		res += int(np.argmax(network.compute(images[i])) == np.argmax(targets[i]))
	return res

def analysis(layers, learningRate, batchSize, iteration, probabilistic):
	print("Download")
	fileNames= ["train-labels-idx1-ubyte", "train-images-idx3-ubyte",
		"t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte"]
	for fileName in fileNames:
		Extraction.downloadDecompress("http://yann.lecun.com/exdb/mnist/",
			fileName, "Data/")

	print("Start extraction")
	training_labels, training_images = Extraction.extractImagesLabels("Data/train-labels-idx1-ubyte", "Data/train-images-idx3-ubyte")
	testing_labels, testing_images = Extraction.extractImagesLabels("Data/t10k-labels-idx1-ubyte", "Data/t10k-images-idx3-ubyte")

	net = Network.NeuralNetwork(layers)

	print("Start backpropagation")
	net.backpropagation(training_images, training_labels, learningRate, batchSize,
		probabilistic, iteration)

	print("Test")
	print("On the training set : {} / {}".format(
		test(net, training_images, training_labels), len(training_labels)))
	print("On the testing set : {} / {}".format(
		test(net, testing_images, testing_labels), len(testing_labels)))

if __name__ == '__main__':
	analysis([784,50,10],0.01,10,20,True)
