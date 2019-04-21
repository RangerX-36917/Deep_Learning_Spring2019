from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
class torchMLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(torchMLP, self).__init__()
        self.linear_layers = []
        self.linear_layers.append(nn.Linear(n_inputs, n_hidden[0]))
        for i in range (len(n_hidden) - 1):
            self.linear_layers.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
        self.linear_layers.append(nn.Linear(n_hidden[-1], n_classes))
        self.linear_layers = nn.ModuleList(self.linear_layers)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for i in range(len(self.linear_layers) - 1):
            x = F.relu(self.linear_layers[i].forward(x))
        return F.softmax(self.linear_layers[-1].forward(x))
