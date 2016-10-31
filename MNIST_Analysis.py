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

minZero = 0.00000000000001
maxOne = 0.9999999999999

zero = np.array([maxOne,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero]).reshape((10,1))
one = np.array([minZero,maxOne,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero]).reshape((10,1))
two = np.array([minZero,minZero,maxOne,minZero,minZero,minZero,minZero,minZero,minZero,minZero]).reshape((10,1))
three = np.array([minZero,minZero,minZero,maxOne,minZero,minZero,minZero,minZero,minZero,minZero]).reshape((10,1))
four = np.array([minZero,minZero,minZero,minZero,maxOne,minZero,minZero,minZero,minZero,minZero]).reshape((10,1))
five = np.array([minZero,minZero,minZero,minZero,minZero,maxOne,minZero,minZero,minZero,minZero]).reshape((10,1))
six = np.array([minZero,minZero,minZero,minZero,minZero,minZero,maxOne,minZero,minZero,minZero]).reshape((10,1))
seven = np.array([minZero,minZero,minZero,minZero,minZero,minZero,minZero,maxOne,minZero,minZero]).reshape((10,1))
eight = np.array([minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,maxOne,minZero]).reshape((10,1))
nine = np.array([minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,maxOne]).reshape((10,1))

def test(network, images, targets):
	number = 0
	res = []
	for i in range(len(images)):
		imageRes = network.compute(images[i])
		res.append(imageRes)
		number += int(np.argmax(imageRes) == np.argmax(targets[i]))
	return number, np.array(res)

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

	numberTrain, outputTrain = test(net, training_images, training_labels)
	numberTest, outputTest = test(net, testing_images, testing_labels)
	print("Test")
	print("On the training set : {} / {}".format(numberTrain, len(training_labels)))
	print("On the testing set : {} / {}".format(numberTest, len(testing_labels)))

	return net, outputTrain, outputTest

def computeReverseImage(net, target, imageFile, x = 28, y = 28):
	image = net.reverseCompute(target).reshape(x,y)
	# Reverse extraction phasis
	image = image* 255
	scipy.misc.imsave(imageFile, image)

def computeImage(net, input, imageFile, x = 28, y = 28):
	print(net.compute(input).shape)
	image = net.compute(input).reshape(x,y)
	# Reverse extraction phasis
	image = image* 255
	scipy.misc.imsave(imageFile, image)
