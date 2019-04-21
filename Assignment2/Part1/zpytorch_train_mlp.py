from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics
import argparse
import numpy as np
from pytorch_mlp import PyMLP
import torch.nn as nn
import torch.autograd as autograd
import torch
from sklearn import datasets as sd
import torch.optim as optim
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

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
    accuracy = metrics.accuracy_score(targets, predictions)
    return accuracy

def train(pymlp,x_train, y_train, x_test, y_test, style="sgd", epochs=MAX_EPOCHS_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).long()
    # print("x_train",x_train)
    # print("y_train",y_train)
    train_accuracies = []
    test_accuracies = []
    if style == "sgd":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pymlp.parameters(), lr=LEARNING_RATE_DEFAULT)
        for epoch in range(epochs):
            ouput = None
            loss = None
            for xi, yi in zip(x_train, y_train):
                optimizer.zero_grad()
                output = pymlp(xi).view(1,-1)
                # print(output)
                target = yi.view(1,-1)[0]

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Accuracy
            output = pymlp(torch.Tensor(x_train))
            # print(output)
            output = to_class_index(output)
            # print(output)
            acc = accuracy(y_train, output)
            print('Epoch [%d/%d]Loss: %.4f Acc: %.3f' % (epoch + 1, epochs, loss.data, acc))

    return train_accuracies, test_accuracies


def to_class_index(output):
    prediction = []
    for i in output:
        prediction.append((i[0] < i[1]))
    return torch.Tensor(prediction).long()


def getdata():
    x, y = sd.make_moons(1000, noise=0.01)
    # view of the data
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, label="full data set")
    # plt.legend()
    # plt.show()
    # y = onehot_trans(y)

    # seperate the training data and test data
    sep = (int)(0.8 * len(x))
    x_train = x[:sep,:]
    x_test = x[sep:,:]
    y_train = y[:sep]
    y_test = y[sep:]
    return x_train, x_test, y_train, y_test


def main():
    """
    Main function
    """
    pymlp = PyMLP(2,[20],2)
    x_train, x_test, y_train, y_test = getdata()
    train(pymlp, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
