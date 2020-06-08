'''
Created on 13 mar. 2017

@author: carlos
'''

import abc


class Kernel(object):
    '''
    Generic Kernel class
    '''
    __metaclass__ = abc.ABCMeta
    _jitter = 1e-10
    
    def __init__(self, jitter=1e-10):
        self._jitter = jitter
   
    @abc.abstractmethod
    def kernel(self, X, X2=None):
        raise NotImplementedError("Subclass should implement this.")
    
    
    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")
        
    @classmethod
    def jitter(self):
        return self._jitter
