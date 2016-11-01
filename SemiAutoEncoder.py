"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import NeuralNetwork.Save as save
import MNIST_Analysis as mnist

if __name__ == '__main__':
	trainL, trainI, testL, testI = mnist.dataExtraction()

	decoder, outTrain, outTest = mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI,testL,testI, "SemiAutoDecoder.pkl")
	encoder, outTrain2, outTest2 = mnist.analysis([10,25,784],0.01,10,10,True,trainI,outTrain,testI, outTest, "SemiAutoEncoder.pkl")

	testNet = save.load("SemiAutoEncoder.pkl")

	mnist.computeImage(testNet, mnist.zero, "Images/e0.zero.png")
	mnist.computeImage(testNet, mnist.one, "Images/e1.one.png")
	mnist.computeImage(testNet, mnist.two, "Images/e2.two.png")
	mnist.computeImage(testNet, mnist.three, "Images/e3.three.png")
	mnist.computeImage(testNet, mnist.four, "Images/e4.four.png")
	mnist.computeImage(testNet, mnist.five, "Images/e5.five.png")
	mnist.computeImage(testNet, mnist.six, "Images/e6.six.png")
	mnist.computeImage(testNet, mnist.seven, "Images/e7.seven.png")
	mnist.computeImage(testNet, mnist.eight, "Images/e8.eight.png")
	mnist.computeImage(testNet, mnist.nine, "Images/e9.nine.png")

	mnist.computeImage(encoder, (mnist.nine + mnist.two)/2., "Images/e92.png")
	mnist.computeImage(encoder, (mnist.five + mnist.three)/2., "Images/e53.png")
