import numpy as np

from . node import *

class Tensor(Node):
    def __init__(self, shape: list, init=False, trainable=True, **kargs) -> None:

        Node.__init__(self, **kargs)
        

        if init: self.outputs = np.mat(np.random.normal(0, 0.001, shape)) 
        #else: self.outputs = np.mat(np.zeros(shape))
        #self.numOfElements = np.prod(shape)
        self.shape = shape
        self.trainable = trainable


    def set_value(self, value):
        #print(value.shape)
        #print(self.shape())
        assert isinstance(value, np.matrix) and value.shape == self.shape
        
        self.reset_value()
        self.outputs = value