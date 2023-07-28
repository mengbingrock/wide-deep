
from .node import Node
import numpy as np
import abc


class Metrics(Node):
    '''
    评估指标算子抽象基类
    '''

    def __init__(self, *parents, **kargs):
        # 默认情况下，metrics节点不需要保存
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)
        # 初始化节点
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    @abc.abstractmethod
    def init(self):
        # 如何初始化节点由具体子类实现
        pass

    @staticmethod
    def prob_to_label(prob, thresholds=0.0):
        if prob.shape[0] > 1:
            # 如果是多分类，预测类别为概率最大的类别
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            # 否则以thresholds为概率阈值判断类别
            #labels = np.where(prob < thresholds, -1, 1)
            labels = np.where(prob < thresholds, 0, 1)

        return labels

    def get_jacobi(self):

        # 对于评估指标节点，计算雅可比无意义
        raise NotImplementedError()

    def value_str(self):
        print(self.__class__.__name__, self.outputs)
        return "{}: {:.4f} ".format(self.__class__.__name__, self.outputs)


class Accuracy(Metrics):
    '''
    正确率节点
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        '''
        计算Accrucy: (TP + TN) / TOTAL
        这里假设第一个父节点是预测值（概率），第二个父节点是标签
        '''

        pred = Metrics.prob_to_label(self.parents[0].outputs) * 2 - 1
        gt = self.parents[1].outputs
        print('pred=', self.parents[0].outputs, pred, 'gt=', gt)
        assert len(pred) == len(gt)
        if pred.shape[0] > 1:
            self.correct_num += np.sum(np.multiply(pred, gt))
            self.total_num += pred.shape[1]
        else:
            self.correct_num += np.sum(pred == gt)
            self.total_num += len(pred)
        self.outputs = 0
        if self.total_num != 0:
            self.outputs = float(self.correct_num) / self.total_num