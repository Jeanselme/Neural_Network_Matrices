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
	net, outTrain, outTest = mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI,testL,testI)
	mnist.computeReverseImage(net, mnist.zero, "Images/0.zero.png")
	mnist.computeReverseImage(net, mnist.one, "Images/1.one.png")
	mnist.computeReverseImage(net, mnist.two, "Images/2.two.png")
	mnist.computeReverseImage(net, mnist.three, "Images/3.three.png")
	mnist.computeReverseImage(net, mnist.four, "Images/4.four.png")
	mnist.computeReverseImage(net, mnist.five, "Images/5.five.png")
	mnist.computeReverseImage(net, mnist.six, "Images/6.six.png")
	mnist.computeReverseImage(net, mnist.seven, "Images/7.seven.png")
	mnist.computeReverseImage(net, mnist.eight, "Images/8.eight.png")
	mnist.computeReverseImage(net, mnist.nine, "Images/9.nine.png")
