import numpy as np
from numpy.core.fromnumeric import shape
from scipy import signal
from layer import Layer
from sklearn.cluster import KMeans
import random 

class Convolutional(Layer):
    def __init__(self, n_filters, input_shape, kernel_size = 3, Mfilter_shape = (4, 3, 3), depth = 4):
        
        self.input_shape = input_shape
        self.n_filters = n_filters
        batch_size = input_shape[0]
        input_depth =  input_shape[1]
        input_height =  input_shape[2]
        input_width = input_shape[3]
        
        self.depth = depth
        self.batch_size = batch_size
        
        self.input_depth = input_depth

        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        Mfilter_depth, Mfilter_height, Mfilter_width = Mfilter_shape
        self.Mfilter_shape = Mfilter_shape

        
        self.reconstructed= np.zeros((self.n_filters, self.batch_size,  Mfilter_depth,Mfilter_depth,Mfilter_height,Mfilter_width))
        self.reconstructed_shape= (self.n_filters, self.batch_size, Mfilter_depth,Mfilter_depth,Mfilter_height,Mfilter_width)

        self.kernels_unbinarized= np.random.randn(*( self.n_filters, self.batch_size, input_depth,kernel_size,kernel_size))
        self.kernels_binarized= np.zeros((self.n_filters, self.batch_size, input_depth,kernel_size,kernel_size))

        self.biases = np.random.randn(*self.output_shape)
        self.Mfilter = np.random.randn(*( self.n_filters, self.batch_size, input_depth,kernel_size,kernel_size))
    
    def binarised_filter(self):
        for g in range(len(self.kernels_unbinarized)):
            for h in range(len(self.kernels_unbinarized[g])):
                k,l,w = self.kernels_unbinarized[g, h].shape
                for c in range (k):
                    k_means = KMeans(init = "k-means++", max_iter = 10,n_clusters = 2, n_init = 10)
                    k_means.fit(self.kernels_unbinarized[g, h, c].reshape(-1, 1))
                    k_means_cluster_centers = k_means.cluster_centers_
                    a1 = k_means_cluster_centers[0]
                    a2 = k_means_cluster_centers[1]
                    for i in range(l):
                        for j in range(w):
                            if np.abs(self.kernels_unbinarized[g, h, c,i,j] - a1) < np.abs(self.kernels_unbinarized[g, h, c,i,j] - a2):
                                self.kernels_binarized[g, h, c,i,j] = a1
                            else:
                                self.kernels_binarized[g, h, c,i,j] = a2

    

    def reconstructed_filter(self):
        for g in range(len(self.Mfilter)):
            for h in range(len(self.Mfilter[g])):
                m_prime = np.array([self.Mfilter[g, h], self.Mfilter[g, h], self.Mfilter[g, h], self.Mfilter[g, h]])
                for i in range(len(self.kernels_unbinarized[g, h])):
                    for m in range(len(m_prime)):
                        for j in range(len(m_prime[m])):
                            self.reconstructed[g, h, i,m,:,:] += self.kernels_unbinarized[g, h, i]*m_prime[m,j]

    def Mcon(self, input,padding=0,stride=1):
        # calcul des filtres
        self.binarised_filter()
        self.reconstructed_filter()
    

        h = (input.shape[2] -  self.reconstructed.shape[5] + 2*padding+1)//stride 
        w = (input.shape[3] -  self.reconstructed.shape[5] + 2*padding+1)//stride
        c = self.reconstructed.shape[3]

        output = np.zeros((self.n_filters, c, h, w))
        print("output shape : ", output.shape)
        for i in range(len( self.reconstructed)):
            for j in range(len(self.reconstructed[i])): 
                for ch in range(len(input[j])):
                    for b in range(len(self.reconstructed[i,j, ch])):
                        output[j, ch] +=  signal.convolve2d(input[j, ch], self.reconstructed[i,j, ch, b], mode='valid')
        return output
    
    def softmax_loss(self, y_true, y_pred):
        loss = - np.sum(y_true*np.math.log(y_pred)) 
        return loss

    def M_filter_loss(self, theta):
        loss = 0
        for c in range(len(self.kernels_unbinarized)):
            loss += (theta/2) * np.linalg.norm(self.kernels_unbinarized[c] - self.reconstructed[0, c, :, :])**2
        return loss


    def forward(self, input):
        return self.Mcon(input)



    def backward(self, output_gradient_ls, theta, learning_rate):
        #reconstructed_gradient = np.zeros(self.reconstructed_shape)
        dLs_dQ = np.zeros(self.reconstructed.shape)
        dLs_dc = np.zeros(self.kernels_unbinarized.shape)
        dLs_dm = np.zeros(self.kernels_unbinarized.shape)
        
        m_prime = np.zeros(self.Mfilter.shape)
        
        input_gradient = np.zeros(self.input_shape)
        Mfilter_gradient = np.zeros(self.Mfilter_shape)
        unbinarized_grad=np.zeros(self.unbinarized_shape)
        bin_grad = np.zeros(self.kernels_binarized.shape)
        
        for g in range(len(self.Mfilter)):
            for h in range(len(self.Mfilter[g])):
                m_prime[g, h] = np.array([self.Mfilter[g, h], self.Mfilter[g, h], self.Mfilter[g, h], self.Mfilter[g, h]])
                

        # Compute gradiant
        for g in range(len(self.kernels_binarized)):
            for h in range(len(self.kernels_binarized[g])):
                for ch in range(len(self.kernels_binarized[g, h])):
                    bin_grad[g, h, ch] = np.gradient(self.kernels_binarized[g, h, ch], axis = 0)
                    bin_grad[g, h, ch] = np.gradient(bin_grad[g, h, ch], axis = 1)

        # unbin grads
        dLm_dc = np. zeros(self.kernels_unbinarized.shape)
        for g in range(len(self.kernels_unbinarized)):
            for h in range(len(self.kernels_unbinarized[g])):
                for ch in range(len(self.kernels_unbinarized[g, h])):
                    dLm_dc[g, h, ch] += theta * ( self.kernels_unbinarized[g, h, ch] - self.reconstructed[g, h, 0, ch])
                    
        for g in range(len(self.reconstructed)):
            for h in range(len(self.reconstructed[g])):
                for i in range(self.reconstructed[g, h, 0]):
                        dLs_dQ[g, h, 0,i] = signal.correlate2d(self.input[g, h, i], output_gradient_ls[i], "valid")
                dLs_dQ[g, h] = np.array([dLs_dQ[g, h, 0,i], dLs_dQ[g, h, 0,i], dLs_dQ[g, h, 0,i], dLs_dQ[g, h, 0,i] ])

        for g in range(len(m_prime)):
            for h in range(len(m_prime[g])):
                for ch in range(len( m_prime[g, h])):
                    for j in range(len( m_prime[g, h, ch])):
                        dLs_dc[g, h, i] += dLs_dQ[g, h, ch, j] * m_prime[g, h, ch, j]
        
        # M_filter grad
        dLm_dm = np.zeros(self.kernels_unbinarized.shape[0], self.kernels_unbinarized.shape[1])
        for g in range(len(self.kernels_unbinarized)):
            for h in range(len(self.kernels_unbinarized[g])):
                for ch in range(len(self.kernels_unbinarized[g, h])):
                    dLm_dm[g, h] += - theta * ( (self.kernels_unbinarized[g, h, ch] - self.reconstructed[g, h, 0, ch]) * self.binarised_filter[g,h, ch] ) 
        for g in range(len( dLs_dQ)):
            for h in range(len( dLs_dQ[g, h])):            
                for i in range(len( dLs_dQ[g, h])):   
                    for j in range(len(dLs_dQ[g, h, i])):
                        dLs_dm[g, h, i] += dLs_dQ[g, h, i, j] * self.kernels_unbinarized[g, h, i]
        
        # Grads
        for g in range(len(self.kernels_unbinarized)):
            for h in range(len(self.kernels_unbinarized[g])):
            
                unbinarized_grad[g, h]= dLs_dc[g, h] +  dLm_dc[g, h] + theta* (self.kernels_binarized[g, h] + learning_rate*bin_grad[g, h] - self.kernels_unbinarized[g, h])
                Mfilter_gradient[g, h] = dLs_dm[g, h] +  dLs_dc[g, h]
                
                self.kernels_unbinarized[g, h] -= learning_rate* unbinarized_grad[g, h]
                self.Mfilter[g, h] = np.abs( self.Mfilter[g, h] - learning_rate* Mfilter_gradient[g, h])

        # Updating filters
        self.reconstructed_filter()





        


    



    """def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient"""