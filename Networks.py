"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import MNIST_Analysis as mnist

if __name__ == '__main__':
	trainL, trainI, testL, testI = mnist.dataExtraction()

	# Probabilistic
	mnist.analysis([784,25,10],0.001,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.005,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.05,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.1,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.5,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],1,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],1.5,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],2,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],2.5,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],3,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],3.5,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],4,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],4.5,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],5,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],5.5,10,10,True,trainL,trainI, testL, testI)

	# Non probabilistic
	mnist.analysis([784,25,10],0.001,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.005,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.05,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.1,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.5,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],1,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],1.5,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],2,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],2.5,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],3,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],3.5,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],4,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],4.5,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],5,10,10,False,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],5.5,10,10,False,trainL,trainI, testL, testI)

	# Iteration
	mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,10,20,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,10,30,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,10,40,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,10,50,True,trainL,trainI, testL, testI)

	# Batch size
	mnist.analysis([784,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,50,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,25,10],0.01,100,10,True,trainL,trainI, testL, testI)

	# Deep
	mnist.analysis([784,100,50,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,100,25,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,300,200,100,10],0.01,10,10,True,trainL,trainI, testL, testI)

	# High hidden links
	mnist.analysis([784,200,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,300,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,300,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,500,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,600,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,700,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,800,10],0.01,10,10,True,trainL,trainI, testL, testI)
	mnist.analysis([784,900,10],0.01,10,10,True,trainL,trainI, testL, testI)
