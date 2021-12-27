import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, n_units , input_size):
        self.units = int(n_units)
        self.weights = np.random.randn(input_size[0], self.units, input_size[1])
        self.bias = np.random.randn(input_size[0], self.units , 1)

    def forward(self, input):
        input = np.reshape(input, (input.shape[0], input.shape[1], 1))
        output = np.zeros((input.shape[0], self.units, 1 ))

        self.input = input

        for i in range(len(input)):
            output[i] = np.dot(self.weights[i], self.input[i]) + self.bias[i]
        
        return np.reshape(output, (input.shape[0], self.units))

    def backward(self, output_gradient, theta, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
