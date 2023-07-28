import numpy as np
import abc

import sys
sys.path.append('..')

from graph.graph import default_graph

class Node:
    def __init__(self, *parents, **kargs):
        
        self.parents = list(parents)
        self.kargs = kargs

        if 'graph' not in kargs:
            self.graph = default_graph
        else:
            self.graph = kargs['graph']
        #self.name = self.gen_node_name(**kargs)
        self.name = self.__class__.__name__ + ':' +self.graph.node_count().__str__()
        print('Node name: {}'.format(self.name))
        self.need_save = kargs.get('need_save', True)
        
        
        
        self.childs = []
        self.outputs = None
        self.jacobian = None

        for parent in self.parents:
            parent.childs.append(self)
        
        self.graph.add_node(self)

    def gen_node_name(self, **kargs):
        """
        生成节点名称，如果用户不指定，则根据节点类型生成类似于"MatMul:3"的节点名，
        如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
        """
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))

    def forward(self):
        for parent in self.parents:
            if parent.outputs is None:
                parent.forward()
        
        self.compute()

    def shapes(self):
        return self.outputs.shape

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值
        """

    @abc.abstractmethod
    def get_jacobian(self, parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵
        """


    def clear_jacobian(self):
        self.jacobian = None


    def dimension(self):
        """
        返回本节点的值展平成向量后的维数
        """
        return self.outputs.shape[0] * self.outputs.shape[1]
    
    def get_childs(self):
        """
        获取本节点的子节点
        """
        return self.childs
    

    def backward(self, result):
        print('backward', self.name)
        if self.jacobian is None:
            if self is result:
                #print('self == result')
                #print(self.name, result.name)
                self.jacobian = np.mat(np.eye(self.dimension()))
            else:
                self.jacobian = np.mat(
                    np.zeros((result.dimension(), self.dimension())))
                #print(result.name, self.name)
                #print('backward dimensions', result.dimension(), self.dimension())

                for child in self.get_childs():
                    if child.outputs is not None:
                        
                        self.jacobian += child.backward(result) * child.get_jacobian(self)
                        #print('===',self.name,child.name)
                        #print(self.jacobi)

        return self.jacobian
    
    def reset_value(self, recursive = True):
        self.outputs = None
       
        if recursive:
            for child in self.get_childs():
                child.reset_value()


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