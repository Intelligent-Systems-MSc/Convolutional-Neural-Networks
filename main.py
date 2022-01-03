import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid, Softmax, Tanh
from losses import binary_cross_entropy, binary_cross_entropy_prime,mse_prime,mse,cross_entropy,cross_entropy_prime
from network import train, predict

def copy4 (x_tr):
    x_tr_prime = np.zeros((x_tr.shape[0], 4, x_tr.shape[1], x_tr.shape[2] ))
    for i in range(len(x_tr_prime)):
        x_tr_prime[i] = np.array([x_tr[i], x_tr[i], x_tr[i], x_tr[i] ])
    return x_tr_prime


def preprocess_data(x, y, limit):
    x = x[:limit]
    y = y[:limit]
    #x = x.reshape(limit, 4, 28, 28)
    x = x.astype("float32") / 255
    x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = copy4(x_train)
x_test = copy4(x_test)

x_train, y_train = preprocess_data(x_train, y_train, 150)
x_test, y_test = preprocess_data(x_test, y_test, 150)

network = [
    Convolutional((1,4, 28, 28), 3, 10),#20*4*26*26
    Tanh(),
    Convolutional((10,4,26,26),3, 20),
    Tanh(),
    Convolutional((20,4,24,24),3, 40),
    Tanh(),
    Reshape((40,4,22, 22), (40*4*22*22,1)),
    Dense(40*4*22*22, 100),
    Tanh(),
    Dense(100, 10),
    Softmax()
]

# train
print("ok")
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=100,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    #print("pred: %f), true: %f",np.argmax(output),np.argmax(y))
