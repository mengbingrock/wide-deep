
from .node import Node
import numpy as np


class LossFunction(Node):
    pass


class PerceptionLoss(LossFunction):

    def compute(self):
        
        self.outputs = np.mat(np.where(self.parents[0].outputs >=0.0, 0.0, -self.parents[0].outputs))
        

    
    def get_jacobian(self, parent):
        diag = np.where(parent.outputs >= 0, 0.0, -1.0)

        return np.diag(diag.ravel())
    

