import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def softmax(self, input):
        out = np.exp(input)
        return out/np.sum(out)
        
class relu(Activation):
    def __init__(self):
        def relu(x):
            y = np.zeros(x.shape)
            for i in range(len(x)):
                    for c in range(len(x[i])):
                                y[i, c] = np.where(x[i,c] <= 0, 0, x[i,c])
            return y

        def relu_prime(x):
            y = np.zeros(x.shape)
            for i in range(len(x)):
                for c in range(len(x[i])):
                        y[i, c] = np.where(x[i,c] <= 0, 0, 1)
            return y

        super().__init__(relu, relu_prime)
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
