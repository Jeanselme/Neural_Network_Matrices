# Neural_Network_Matrices
Neural network with backpropagation

## BackPropagation
This neural network is based on the book : http://neuralnetworksanddeeplearning.com/

## Execution
```
python3.5 MNIST_Analysis.py
```

## Results
The following graph shows the recognition rate for a neural network of 25 hidden neurons, after ten iterations of the backpropagation with a batch of ten images.  
Blue : Probabilistic  
Red : Non Probabilistic  
![Result](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Matrices/master/Images/Recognition-LearningRate.png)

We observe few differences between both. That could explain by the fact that the set is already shuffled. However it could be interesting to train the network with the data which add the more information. Those data are the ones which have the most important error on the previous epoch.

## Libraries
Needs struct, urllib.request, io, gzip, numpy and os. Executed with python3.5
