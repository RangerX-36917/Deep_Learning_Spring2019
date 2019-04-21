from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.lRate = 1e-2
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = []
        #input layer
        n1 = n_inputs
        n2 = n_hidden[0]
        for i in range(1,len(n_hidden)):
            self.layers.append(Linear(n1, n2))
            self.layers.append(ReLU())
            #hidden layer
            n1 = n_hidden[i - 1]
            n2 = n_hidden[i]
        self.layers.append(Linear(n1, n2))
        self.layers.append(ReLU())
        # output layer
        self.layers.append(Linear(n2,n_classes))
        self.layers.append(SoftMax())
        '''
        for i in range(len(self.layers)):
            if(i%2 == 0):
                print(self.layers[i].weight)
            else:
                print("ReLU or Softmax")
        '''
    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        _input = x
        for i in range(len(self.layers)):
            _output = self.layers[i].forward(_input)
            _input = _output
        return _output

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        for i in range(len(self.layers) - 1, -1, -1):
            dout = self.layers[i].backward(dout)
            
        return dout
