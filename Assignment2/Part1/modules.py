import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.weight = np.random.normal(0,0.1,size = (out_features, in_features))
        self.bias = np.zeros(out_features)
        self.grad = []
        
    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.n_input = len(x)
        self.x = x
        out = np.dot(self.weight,x) + self.bias
        self.n_output = len(out)
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        
        self.dw = np.zeros((self.n_output,self.n_input))
        self.db = np.reshape(dout,(1,len(dout)))[0]
        dout = np.reshape(dout, (len(dout),1))
        self.x = np.reshape(np.array(self.x),(len(self.x),1))
        self.dw = np.multiply(self.x.T,dout)
        return np.dot(self.weight.T, dout)

class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x
        self.out = [max(x[i],0) for i in range(len(x))]
        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        self.dx = np.zeros(len(self.x))
        for i in range(len(self.x)):
            for j in range(len(dout)):
                if(self.out[i] >= 0):
                    self.dx[i] = dout[i]
        return self.dx

class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        #this code is from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        self.x = x
        b = x.max()
        y = np.exp(x - b)
        self.out = y / y.sum()
        return self.out

    def backward(self, label):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        """
        _dx = np.zeros((len(self.x),1))
        for i in range(len(self.x)):
            for j in range(len(dout)):
                if(i == j):
                    _dx[i] = _dx[i] + self.out[i] * (1-self.out[j]) * dout[j]
                else:
                    _dx[i] = _dx[i] + (0 - self.out[j] * self.out[i]) * dout[j]
        self.dx = _dx
        """
        return self.out - label

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        e = 0-np.dot(y,np.log(x))
        return e
    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        dx = np.zeros(len(y))
        
        for i in range(len(y)):
            if(x[i] == 0):
                if(y[i] == 0):
                    dx[i] = 0
                else:
                    dx[i] = 100
            else:
                dx[i] = 0 - y[i]/x[i]
        return dx
