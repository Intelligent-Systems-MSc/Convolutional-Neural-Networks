import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        #print(self.input.shape)
        #print(output_gradient.shape)
        #print("-----------")
        x = np.zeros(self.input.shape)
        if (output_gradient.shape != self.input.shape):
            for i_out in range (output_gradient.shape[0]):
                for i_in in range(self.input.shape[0]):
                    #print(output_gradient[i_out])
                    #print(self.input[i_in])
                    x[i_in] += np.multiply(output_gradient[i_out], self.activation_prime(self.input[i_in]))
        else:
            x = np.multiply(output_gradient, self.activation_prime(self.input)) 
        

        return x
