"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import os
import sys
import numpy as np
import scipy.misc
import NeuralNetwork.Network as Network
import DataExtraction.Extraction as Extraction

def test(network, images, targets):
	res = 0
	for i in range(len(images)):
		res += int(np.argmax(network.compute(images[i])) == np.argmax(targets[i]))
	return res

def dataExtraction():
	print("Download")
	fileNames= ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]
	for fileName in fileNames:
		Extraction.downloadDecompress("http://yann.lecun.com/exdb/mnist/",
			fileName, "Data/")

	print("Start extraction")
	training_labels, training_images = Extraction.extractImagesLabels(
		"Data/train-labels-idx1-ubyte.gz", "Data/train-images-idx3-ubyte.gz")
	testing_labels, testing_images = Extraction.extractImagesLabels(
		"Data/t10k-labels-idx1-ubyte.gz", "Data/t10k-images-idx3-ubyte.gz")
	return training_labels, training_images, testing_labels, testing_images

def analysis(layers, learningRate, batchSize, iteration, probabilistic,
	training_labels, training_images, testing_labels, testing_images):
	print("\n" + str(layers) + " - Batch {} - Rate {} - Probabilitic {}".format(
		batchSize, learningRate, probabilistic))
	net = Network.NeuralNetwork(layers)

	print("Start backpropagation")
	net.backpropagation(training_images, training_labels, learningRate, batchSize,
		probabilistic, iteration)

	print("Test")
	print("On the training set : {} / {}".format(
		test(net, training_images, training_labels), len(training_labels)))
	print("On the testing set : {} / {}".format(
		test(net, testing_images, testing_labels), len(testing_labels)))

	return net

def computeImage(net, target, imageFile, x = 28, y = 28):
	image = net.reverseCompute(target).reshape(x,y)
	# Reverse extraction phasis
	image = image* 255
	scipy.misc.imsave(imageFile, image)
