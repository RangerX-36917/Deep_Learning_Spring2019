from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class PyMLP(nn.Module):

    def __init__(self, n_inputs=2, n_hidden=[3], n_classes=2):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(PyMLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.linear_layers = []
        self.linear_layers.append(nn.Linear(n_inputs, n_hidden[0]))
        for i in range(len(n_hidden) - 1):
            self.linear_layers.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
        # building the model
        self.linear_layers.append(nn.Linear(n_hidden[-1], self.n_classes))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        # self.SM = nn.Softmax()


    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        input = x
        for i in range(len(self.linear_layers)-1):
            input = F.relu(self.linear_layers[i].forward(input))
        input = F.softmax(self.linear_layers[-1].forward(input))
        return input

    # def predict(self, x):
    #     """
    #     Predict the result.
    #     Args:
    #         x: input of the network
    #     """
    #     h2 = self.forward(x)
    #     # transfer to one-hot
    #     predict = np.zeros(h2.shape[0])
    #     for i in range(len(h2)):
    #         predict[i] = 1 if h2[i][0] > h2[i][1] else 0
    #     return predict
if __name__ == '__main__':
    mlp = MLP()
    print(mlp)
    xi = torch.Tensor(np.array([1,2])).view(1,-1)
    yi = torch.LongTensor([1])

    criterion = nn.CrossEntropyLoss()
    print(output)
