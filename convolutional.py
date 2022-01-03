import numpy as np
from scipy import signal
from layer import Layer
from sklearn.cluster import KMeans

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        
        #input_dept must be 4
        input_depth, input_height, input_width = input_shape[1:4]
        
        self.n_features = input_shape[0]
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth,input_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth,self.n_features, input_depth, kernel_size, kernel_size)
        
        self.unbinarized_filter = np.random.randn(*(self.kernels_shape))
        self.biases = np.random.randn(*self.output_shape)

        self.M_filter = np.random.randn(*self.kernels_shape)
        
        self.binarized_filter = np.random.randn(*self.kernels_shape)
        
        self.reconstructed_filter = np.random.randn(*(self.depth,self.n_features,self.input_depth,self.input_depth,kernel_size,kernel_size))
    
    def make_binarised_filter(self):
        d,f,k,h,w = self.unbinarized_filter.shape
        #print("ok")
        for d in range(d):
            for f in range(f):
                for c in range (k):
                    k_means = KMeans(init = "k-means++", max_iter = 10,n_clusters = 2, n_init = 10)
                    k_means.fit(self.unbinarized_filter[d,f,c].reshape(-1, 1))
                    k_means_cluster_centers = k_means.cluster_centers_
                    a1 = k_means_cluster_centers[0]
                    a2 = k_means_cluster_centers[1]
                    for i in range(h):
                        for j in range(w):
                            if np.abs(self.unbinarized_filter[d,f,c,i,j] - a1) < np.abs(self.unbinarized_filter[d,f,c,i,j] - a2):
                                self.binarized_filter[d,f,c,i,j] = a1
                            else:
                                self.binarized_filter[d,f,c,i,j] = a2
    
    
    def make_reconstructed_filter(self):
        for d in range(self.depth):
            for f in range(self.n_features):
                for j in range(self.input_depth):
                    print(self.M_filter[d,f,j].shape)
                    print(self.binarized_filter[d,f].shape)
                    M_prime = np.array([self.M_filter[d,f,j],self.M_filter[d,f,j],self.M_filter[d,f,j],self.M_filter[d,f,j]])
                    self.reconstructed_filter[d,f,j] = self.binarized_filter[d,f]*M_prime


    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        
        for i in range(self.depth): #nombre de filtres
            for f in range(self.n_features):
                for j in range(self.input_depth): #un plande k de taille 4x3x3 de reconstructed filter
                    for k in range (self.input_depth): #un plan de taille 3x3
                        self.output[i,j] += signal.correlate2d(self.input[f,k], self.reconstructed_filter[i,f,j,k], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        
        kernels_gradient = np.zeros((self.depth,self.n_features,self.input_depth,self.input_depth,self.kernel_size,self.kernel_size))
        input_gradient = np.zeros((self.depth,self.input_depth,self.input_shape[2],self.input_shape[3]))

        
        for n in range(self.depth): # #nombre de features map de taille 4x3x3
            for f in range(self.n_features):
                for i in range(self.input_depth): #parcourir les plans sortie (4)
                    for j in range(self.input_depth): #parcourir les plans de entree
                        
                        kernels_gradient[n,f,i, j] = signal.correlate2d(self.input[f,j], output_gradient[n,i], "valid")
                        
                        input_gradient[n,j] += signal.convolve2d(output_gradient[n,i], self.reconstructed_filter[n,f,i, j], "full")

        dLs_dCi = np.zeros((self.depth,self.n_features,self.input_depth,self.kernel_size,self.kernel_size))
        for d in range(self.depth):
            for f in range(self.n_features):
                for i in range(self.input_depth): 
                    for j in range(self.input_depth):
                        
                        M_prime = np.array([self.M_filter[d,f,j],self.M_filter[d,f,j],self.M_filter[d,f,j],self.M_filter[d,f,j]])
                        dLs_dCi[d,f,i] += np.dot(kernels_gradient[d,f,i,j],M_prime[j])
                       
        
        dLs_dM = np.zeros((self.depth,self.n_features,self.input_depth,self.kernel_size,self.kernel_size))
        for d in range(self.depth):
            for f in range(self.n_features):
                for i in range(self.input_depth): 
                    for j in range(self.input_depth):
                        dLs_dM[d,f,i] += np.dot(kernels_gradient[d,f,i,j],self.binarized_filter[d,f,j])


        
        self.unbinarized_filter = self.unbinarized_filter - 0.1*dLs_dCi
        self.M_filter = self.M_filter - 0.1*dLs_dM
        
        self.biases -= learning_rate * output_gradient
        return input_gradient
