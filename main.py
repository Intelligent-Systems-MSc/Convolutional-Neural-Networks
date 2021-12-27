import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

import matplotlib.pyplot as plt
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from BatchNormalisation import BatchNormalisation
from MaxPool import MaxPool
from activations import Sigmoid, relu
from reshape import Reshape
from losses import softmax_loss, softmax_loss_prime
from network import train, predict

def copy4 (x_tr):
    x_tr_prime = np.zeros((x_tr.shape[0], 4, x_tr.shape[1], x_tr.shape[2] ))
    for i in range(len(x_tr_prime)):
        x_tr_prime[i] = np.array([x_tr[i], x_tr[i], x_tr[i], x_tr[i] ])
    return x_tr_prime

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train= x_train.astype('float32')/float(x_train.max())
x_test= x_test.astype('float32')/float(x_test.max())
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


#neural network
network = [
    Convolutional(20, (10,4,28,28)),
    BatchNormalisation(),
    MaxPool(),
    relu(),
    
    Convolutional(40, (20,4,13,13)),
    BatchNormalisation(),
    MaxPool(),
    relu(),


    Reshape((40,4,5,5), (40,4*5*5)),
    Dense(1024, (40,4*5*5)),
    Dense(10, (40,1024)),
    Sigmoid()
]

# train
# test
epochs = 1
batch_size = 64
nb_batch = int(x_train.shape[0]/batch_size)
theta = 0.001
learning_rate = 0.001
verbose = True

x = x_train[0:10]


for e in range(epochs):
        error = 0
        for i in range(nb_batch + 1):
            if i == nb_batch:
                batch = x_train[i*batch_size: (i+1)*batch_size + (x_train.shape[0] - nb_batch*batch_size)]
                y = y_train[i*batch_size: (i+1)*batch_size + (x_train.shape[0] - nb_batch*batch_size)]
            else:
                batch =x_train[i*batch_size: (i+1)*batch_size]
                y =y_train[i*batch_size: (i+1)*batch_size]
            
            # copy
            
            batch = copy4(batch)
            # forward
            output = predict(network, batch)

            # error
            error += softmax_loss(y, output)

            # backward
            """grad = softmax_loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, theta ,learning_rate)"""

        error /= len(batch)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
            
    
