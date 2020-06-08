'''
Created on 13 mar. 2017

@author: carlos
'''

import tensorflow as tf
from gp_input_noise.kernel import Kernel
        
class RBF_ARD(Kernel):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    
    def __init__(self, log_lengthscales, log_sigma0, log_sigma, jitter = 1e-3):
            
        super(RBF_ARD, self).__init__(jitter)
        
        self.log_lengthscales = tf.Variable(log_lengthscales, dtype = tf.float32)
        self.log_sigma0 = tf.Variable([ log_sigma0 ], dtype = tf.float32)
        self.log_sigma = tf.Variable([ log_sigma ], dtype = tf.float32)
        self.jitter = tf.constant([ jitter ], dtype = tf.float32)

    def kernel(self, X, X2 = None):

        """
        This function computes the covariance matrix for the GP
        """

        if X2 is None:
            X2 = X
            white_noise = (self.jitter + tf.exp(self.log_sigma0)) * tf.eye(tf.shape(X)[ 0 ], dtype = tf.float32)
        else:
            white_noise = 0.0
            
                       
        X = X / tf.sqrt(tf.exp(self.log_lengthscales))
        X2 = X2 / tf.sqrt(tf.exp(self.log_lengthscales))

        value = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        value2 = tf.expand_dims(tf.reduce_sum(tf.square(X2), 1), 1)
        distance = value - 2 * tf.matmul(X, tf.transpose(X2)) + tf.transpose(value2)
        
        return tf.exp(self.log_sigma) * tf.exp(-0.5 * distance) + white_noise

    def get_params(self):
        return [ self.log_lengthscales, self.log_sigma, self.log_sigma0 ]

    def get_log_sigma(self):
        return self.log_sigma

    def get_log_sigma0(self):
        return self.log_sigma0

    def get_var_points(self, data_points):
        return  tf.ones([ tf.shape(data_points)[ 0 ] ]) * tf.exp(self.log_sigma) + (self.jitter + tf.exp(self.log_sigma0))

 

