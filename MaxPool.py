from layer import Layer
import numpy as np


class MaxPool(Layer):
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, a):
        out = np.zeros((a.shape[0], a.shape[1], int(a.shape[2]//2), int(a.shape[3]//2)))
        for b in range(len(a)):
            for c in range(len(a[b])):
                for i in range(int(len(a[b,c])//2)):
                    for j in range(int(len(a[b,c, i])//2)):
                        out[b, c, i, j] =  np.max(a[b,c, 2*i:2*(i +1),2*j:2*(j +1) ])
                
        return out

    def backward(self, output_gradient,theta,  learning_rate):
        # TODO: update parameters and return input gradient
        pass