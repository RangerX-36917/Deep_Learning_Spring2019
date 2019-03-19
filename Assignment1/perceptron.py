import numpy as np

class Perceptron(object):
    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=0.01):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.maxEpoch = max_epochs
        self.lRate = learning_rate
        self.w = np.zeros(n_inputs)
        
    def forward(self, input_data):
        """
        Predict label from input
        Args:
            input: array of dimension equal to n_inputs.
        """
        
        #print("forward", input_data)
        return  np.sign(np.dot(self.w,input_data[0]))
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        training_data = []
        for i in range(len(labels)):
            training_data.append([training_inputs[i],labels[i]])
        #np.random.shuffle(training_data)
        for data in training_data:
            #print(self.w," . ",data[0])
            #print(np.dot(self.w, data[0]), data[1])
            if(np.dot(self.w,data[0])*data[1])<=0:
             #   print(self.w, data[0])
                self.w += np.dot((self.lRate * data[1]), data[0])
              #  print("after:", self.w)
                

