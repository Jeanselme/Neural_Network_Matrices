"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import MNIST_Analysis as mnist
import numpy as np

if __name__ == '__main__':
	trainL, trainI, testL, testI = mnist.dataExtraction()

	# Probabilistic
	net = mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI,testL,testI)
	minZero = 0.00000000000001
	maxOne = 0.9999999999999

	zero = np.array([maxOne,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero])
	one = np.array([minZero,maxOne,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero])
	two = np.array([minZero,minZero,maxOne,minZero,minZero,minZero,minZero,minZero,minZero,minZero])
	three = np.array([minZero,minZero,minZero,maxOne,minZero,minZero,minZero,minZero,minZero,minZero])
	four = np.array([minZero,minZero,minZero,minZero,maxOne,minZero,minZero,minZero,minZero,minZero])
	five = np.array([minZero,minZero,minZero,minZero,minZero,maxOne,minZero,minZero,minZero,minZero])
	six = np.array([minZero,minZero,minZero,minZero,minZero,minZero,maxOne,minZero,minZero,minZero])
	seven = np.array([minZero,minZero,minZero,minZero,minZero,minZero,minZero,maxOne,minZero,minZero])
	eight = np.array([minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,maxOne,minZero])
	nine = np.array([minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,minZero,maxOne])
	mnist.computeImage(net, zero, "Images/0.zero.png")
	mnist.computeImage(net, one, "Images/1.one.png")
	mnist.computeImage(net, two, "Images/2.two.png")
	mnist.computeImage(net, three, "Images/3.three.png")
	mnist.computeImage(net, four, "Images/4.four.png")
	mnist.computeImage(net, five, "Images/5.five.png")
	mnist.computeImage(net, six, "Images/6.six.png")
	mnist.computeImage(net, seven, "Images/7.seven.png")
	mnist.computeImage(net, eight, "Images/8.eight.png")
	mnist.computeImage(net, nine, "Images/9.nine.png")
