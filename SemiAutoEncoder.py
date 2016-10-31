"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import MNIST_Analysis as mnist

if __name__ == '__main__':
	trainL, trainI, testL, testI = mnist.dataExtraction()

	decoder, outTrain, outTest = mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI,testL,testI)
	encoder, outTrain2, outTest2 = mnist.analysis([10,25,784],0.01,10,10,True,trainI,outTrain,testI, outTest)
	mnist.computeImage(encoder, mnist.zero, "Images/e0.zero.png")
	mnist.computeImage(encoder, mnist.one, "Images/e1.one.png")
	mnist.computeImage(encoder, mnist.two, "Images/e2.two.png")
	mnist.computeImage(encoder, mnist.three, "Images/e3.three.png")
	mnist.computeImage(encoder, mnist.four, "Images/e4.four.png")
	mnist.computeImage(encoder, mnist.five, "Images/e5.five.png")
	mnist.computeImage(encoder, mnist.six, "Images/e6.six.png")
	mnist.computeImage(encoder, mnist.seven, "Images/e7.seven.png")
	mnist.computeImage(encoder, mnist.eight, "Images/e8.eight.png")
	mnist.computeImage(encoder, mnist.nine, "Images/e9.nine.png")

	mnist.computeImage(encoder, mnist.nine + mnist.two, "Images/e92.png")
	mnist.computeImage(encoder, mnist.five + mnist.three, "Images/e53.png")
