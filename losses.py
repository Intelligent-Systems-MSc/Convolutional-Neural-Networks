import numpy as np

def mse(y_true, y_pred):
    
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def cross_entropy(y_true, y_pred):
    return np.sum(-y_true*np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    CE_deriv = np.zeros((10,1))
    softmax_deriv=np.zeros((10,1))
    for i in range(len(CE_deriv)):
        CE_deriv[i] = -y_true[i]*(1/y_pred[i])
    for i in range(len(y_true)):
        if (y_true[i] == 1):
            index = i
    #########
    for i in range(len(CE_deriv)): #index = j
        if i == index:
            softmax_deriv[index] = CE_deriv[index]*(1-CE_deriv[index])
        else:
            softmax_deriv[i]= -CE_deriv[i]*CE_deriv[index]
    return softmax_deriv 

########################################

