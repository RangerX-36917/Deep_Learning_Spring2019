from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import argparse
import numpy as np
from  pytorch_mlp import torchMLP
import sklearn.datasets as skd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 20

FLAGS = None


def accuracy(predictions, targets):

    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    correct = 0
    for p, t in zip (predictions, targets):
        if(torch.argmax(p) == t.detach()):
            correct += 1
    accuracy = correct * 100 / len(predictions)
    return accuracy


def train(mlp, dataSet, labels, testDataSet, testLabels, epochs):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    print("train")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE_DEFAULT)
    ep = []
    acs = []
    acs1 = []
    for i in range(epochs):
        index = random.randint(0, len(dataSet) - 1)
        data = dataSet[index]
        label = labels[index]
        optimizer.zero_grad()
        result = mlp.forward(data).view(1, -1)
        label = label.view(1, -1)[0]
        loss = criterion(result, label)
        loss.backward()
        optimizer.step()

        if 0 == i % EVAL_FREQ_DEFAULT:
            results = []
            for data, label in zip(testDataSet, testLabels):
                result = mlp.forward(data)
                results.append(result)
            ac = accuracy(results, testLabels)
            ep.append(i)
            acs.append(ac)
            results = []
            for data, label in zip(dataSet, labels):
                result = mlp.forward(data)
                results.append(result)
            ac = accuracy(results, labels)
            acs1.append(ac)
    return ep, acs, acs1




def main():
    """
    Main function
    """
    # train()
    dataSet, labels = skd.make_moons(1000, shuffle=True, noise=0.01)
    testDataSet, testLabels = skd.make_moons(200, shuffle=True, noise=0.01)

    onehotLabels, features = toOneHot(labels)
    testOnehotLabels, testFeatures = toOneHot(testLabels)

    dataSet = torch.Tensor(dataSet)
    labels = torch.Tensor(labels).long()
    testDataSet = torch.Tensor(testDataSet)
    testLabels = torch.Tensor(testLabels).long()

    mlp = torchMLP(2, [20], 2)

    #train(mlp, dataSet, labels, testDataSet, testLabels, MAX_EPOCHS_DEFAULT)

def toOneHot(labels):
    features = {}
    n_features = 2
    n = 0
    labels_onehot = []
    for l in labels:
        x = np.zeros(n_features)
        if (l in features):
            x[features[l]] = 1
        else:
            features[l] = n
            x[features[l]] = 1
            n += 1
        labels_onehot.append(x)
    return labels_onehot, features
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()




