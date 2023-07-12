import numpy as np
import abc

from .node import Node

def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    pass


class Add(Operator):
    def compute(self):
        self.outputs = np.mat(np.zeros(self.parents[0].shapes()))

        for parent in self.parents:
            self.outputs += parent.outputs

    def get_jacobian(self, parent):
        return np.mat(np.eye(self.dimension()))
    

class MatMul(Operator):
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shapes()[
            1] == self.parents[1].shapes()[0]
        #self.outputs = np.mat(np.zeros((self.parents[0].outputs.shape[0], self.parents[1].outputs.shape[1])))
        self.outputs = self.parents[0].outputs * self.parents[1].outputs

    def get_jacobian(self, parent):
        zeros =  np.mat(np.zeros((self.dimension(), parent.dimension())))

        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].outputs.T)
        
        jacobian = fill_diagonal(zeros, self.parents[0].outputs)
        
        row_idx = np.arange(self.dimension()).reshape(self.shapes()[::-1]).T.ravel()

        col_idx = np.arange(parent.dimension()).reshape(parent.shapes()[::-1]).T.ravel()

        return jacobian[row_idx, :][:, col_idx]
    

class Step(Operator):
    def compute(self):
        self.outputs = np.mat(np.where(self.parents[0].outputs >= 0.0, 1.0, 0.0))

    def get_jacobian(self, parent):
        return np.mat(np.zeros(self.dimension()))
    

