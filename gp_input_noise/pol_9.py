'''
Created on 13 mar. 2017

@author: carlos
'''

import tensorflow as tf
from kernel import Kernel
        
class Pol_9(Kernel):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    
    def __init__(self, log_length_scales, log_sigma0, log_sigma, jitter = 1e-10):
            
        super(Pol_9, self).__init__(jitter)
        
        self.log_length_scales = tf.Variable(log_length_scales, dtype = tf.float32)
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

        X = X / tf.sqrt(tf.exp(self.log_length_scales))
        X2 = X2 / tf.sqrt(tf.exp(self.log_length_scales))
                       
        product = tf.pow((tf.matmul(X, tf.transpose(X2)) + 1.0), 9.0)
        
        return tf.exp(self.log_sigma) * product + white_noise

    def get_params(self):
        return [ self.log_length_scales, self.log_sigma0, self.log_sigma ]

    def get_log_sigma(self):
        return self.log_sigma

    def get_log_sigma0(self):
        return self.log_sigma0

    def get_var_points(self, data_points):

        data_points = data_points / tf.sqrt(tf.exp(self.log_length_scales))

        norms_X = tf.norm(data_points, axis = 1)

        return  tf.pow(norms_X**2 + 1.0, 9.0) * tf.exp(self.log_sigma) + (self.jitter + tf.exp(self.log_sigma0))
 

