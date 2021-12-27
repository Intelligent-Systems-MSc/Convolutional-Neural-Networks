from layer import Layer
import numpy as np

class BatchNormalisation(Layer):
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        return (input - np.mean(input)) / np.std(input) 
        

    def backward(self, output_gradient,theta,  learning_rate):
        pass
