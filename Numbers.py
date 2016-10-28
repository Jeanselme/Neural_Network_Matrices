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

	zero = np.array([0.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
	one = np.array([0.01,0.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
	two = np.array([0.01,0.01,0.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
	three = np.array([0.01,0.01,0.01,0.99,0.01,0.01,0.01,0.01,0.01,0.01])
	four = np.array([0.01,0.01,0.01,0.01,0.99,0.01,0.01,0.01,0.01,0.01])
	five = np.array([0.01,0.01,0.01,0.01,0.01,0.99,0.01,0.01,0.01,0.01])
	six = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.99,0.01,0.01,0.01])
	seven = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.99,0.01,0.01])
	eight = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.99,0.01])
	nine = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.99])
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
