# Neural_Network_Matrices
Neural network with backpropagation

## BackPropagation
This neural network is based on the book : http://neuralnetworksanddeeplearning.com/

## Execution
```
python3.5 Networks.py
```
This script executes the training of several different neural networks in order to evaluate performances.  

```
python3.5 SemiAutoEncoder.py
```
It computes thanks an encoder thanks to the output of a first neuralNetwork and test to draw new images.  

```
python3.5 Numbers.py
```
This one creates the perfect numbers for the network given a specific output.

## Results
The following graph shows the recognition rate for a neural network of 25 hidden neurons, after ten iterations of the backpropagation with a batch of ten images.  
![ResultLearningRate](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Matrices/master/Images/Recognition-LearningRate.png)

We observe few differences between both. That could be explained by the fact that the set is already shuffled. However, it could be interesting to train the network with the data which add the more information. Those data are the ones which have the most important error on the previous epoch.  
Moreover, we notice that for high learning rates, the non probabilistic is worrier than the probabilitic one, contrary to the little learning rates.  

We observe the tendance of the recognition rate in function of the iterations of the backpropagation.  
The increase is obvious, however we observe that it is an augmentation of the difference between the testing set and the training one.
It could be interpreted as an overftting on the trainingset. However it would be neccesary to compute other neural network whith different random weights (because we seed a random value in order to have meaningful comparsion).
![ResultOccurences](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Matrices/master/Images/Recognition-Occurences.png)

When we reverse the computing process in order to observe the "perfect input", we observe the following images :  
![ResultNumbers](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Matrices/master/Images/Numbers.png)

We observe different results with the semi autoencoder (I say semi because it is a two phases supervised learning and not an unsupervided one).  
![ResultNumbersAutoEncoder](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Matrices/master/Images/NumbersSAE.png)
The implemented neural network is the combinaison of two simple neural networks with backpropagation as follows :
![ResultNumbersAutoEncoder](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Matrices/master/Images/SAE.png)

## Libraries
Needs scipy.mnist, struct, urllib.request, io, gzip, numpy and os. Executed with python3.5
