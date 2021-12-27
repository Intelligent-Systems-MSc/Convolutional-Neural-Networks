import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def softmax_loss(y_true, y_pred):
    loss = 0
    for i in range(len(y_true)):
        loss += - np.sum(y_true[i]*np.math.log(y_pred[i])) 
        return np.abs(loss)

def softmax_loss_prime(y_true, y_pred):
    loss = 0
    for i in range(len(y_true)):
        loss += - np.sum(y_true[i]*np.math.log(y_pred[i])) 
        return np.abs(loss)

def M_filter_loss(unbinarized, reconstructed,  theta):
    loss = 0
    for g in range(len(unbinarized)):
        for h in range(len(unbinarized[g])):
            for c in range(len(unbinarized[g,h])):
                loss += (theta/2) * np.linalg.norm(unbinarized[g,h, c] - reconstructed[g, h, 0, c])**2
    return loss
