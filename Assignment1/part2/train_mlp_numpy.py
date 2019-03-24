from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropy
import sklearn.datasets as skd
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
    return accuracy

def train(mlp, dataSet, labels, style):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    if(style == "SGD"):
        j = random.randint(0,len(dataSet)-1)
        data = dataSet[j]
        label = labels[j]
        mlp.forward(np.array(data))
        mlp.backward(label)
        for k in range(len(mlp.layers)):
            if(k%2 == 0):
                mlp.layers[k].weight -=mlp.lRate * mlp.layers[k].dw
                #print(mlp.layers[i].bias, mlp.layers[i].db)
                mlp.layers[k].bias -= mlp.lRate * mlp.layers[k].db
    else:
        w = []
        b = []
        for data, label in zip(dataSet, labels):
            mlp.forward(np.array(data))
            mlp.backward(label)
            for k in range(len(mlp.layers)):
                if(k%2 == 0):
                    if(k >= len(w)):
                        w.append(mlp.layers[k].dw)
                        b.append(mlp.layers[k].db)
                    else:
                        w[k] += mlp.layers[k].dw
                        b[k] += mlp.layers[k].db
                else:
                    if(k/2 > len(w)):
                        w.append(0)
                        b.append(0)
        #print("finish collecting")
        for i in range(len(w)):
            w[i] = w[i]/len(dataSet)
            b[i] = b[i]/len(dataSet)
        
        for k in range(len(mlp.layers)):
            if(k%2 == 0):
                mlp.layers[k].weight -= mlp.lRate*w[k]
                mlp.layers[k].bias -= mlp.lRate*b[k]
       # print("finish update")
            
                


def main():
    """
    Main function
    """
    #train()

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
    dataSet, labels = skd.make_moons(1000, shuffle = True, noise = 0.001)
    testDataSet, testLabels = skd.make_moons(200, shuffle = True, noise = 0.001)
    #print(dataSet, len(dataSet))
    features = {}
    n_features = 2
    n = 0
    labels_onehot = []
    for l in labels:
        x = np.zeros(n_features)
        if(l in features):
            x[features[l]] = 1
        else:
            features[l] = n
            x[features[l]] = 1
            n += 1
        labels_onehot.append(x)

    DNN_HIDDEN_UNITS_DEFAULT = '20'
    LEARNING_RATE_DEFAULT = 1e-1
    MAX_EPOCHS_DEFAULT = 1500
    EVAL_FREQ_DEFAULT = 20
    mlp = MLP(2,[20],2)
    mlp.lRate = LEARNING_RATE_DEFAULT
    CE = CrossEntropy()

    for i in range(MAX_EPOCHS_DEFAULT):
        if(i % EVAL_FREQ_DEFAULT == 0):
            #print(mlp.layers[0].weight)
            correct = 0
            for data, label in zip(testDataSet, testLabels):
                result = mlp.forward(data)
                #print(result, data, label)
                if(np.argmax(result) == features[label] and result.max() != 0.5):
                    correct += 1
            print("test:",i,":",correct*100/len(testDataSet),"%")
            correct = 0
            for data, label in zip(dataSet, labels):
                result = mlp.forward(data)
                #print(result, data, label)
                if(np.argmax(result) == features[label] and result.max() != 0.5):
                    correct += 1
            print("train:",i,":",correct*100/len(dataSet),"%")
        train(mlp, dataSet, labels_onehot, "SGD", 1500)
        
'''
            result = mlp.forward(data)
            #print(result)
            if(result.max() == 0.5):
                continue
            #if(np.argmax(result) != np.argmax(label)):
            dx = CE.backward(result, label)
            mlp.backward(dx)
        ''' 
        
        
    
