# -*- coding: utf-8 -*-

"""
Created on Wed Jul 10 15:19:42 CST 2019

@author: chenzhen
"""
import sys
sys.path.append('..')
import abc
import time

import numpy as np

from node.node import Tensor 
#, default_graph
from .dist.ps import ps
from .dist.allreduce import allreduce

import math
import threading
import time



from graph.graph import (default_graph, get_trainable_variables_from_graph,
                    update_node_value_in_graph)




class Trainer(object):
    '''
    训练器
    '''

    def __init__(self, input_x, input_y,
                 loss_op, optimizer,
                 epoches, batch_size=8,
                 eval_on_train=False, metrics_ops=None, *args, **kargs):

        # 计算图的输入节点，可以有多个，因此类型是list
        self.inputs = input_x

        # 计算图的标签节点
        self.input_y = input_y

        # 损失函数
        self.loss_op = loss_op

        # 优化器
        self.optimizer = optimizer

        # 训练执行的epoch数
        self.epoches = epoches
        self.epoch = 0

        # 批大小
        self.batch_size = batch_size

        # 是否在训练迭代中进行评估
        self.eval_on_train = eval_on_train

        # 评估指标列表
        self.metrics_ops = metrics_ops

        self.print_iteration_interval = kargs.get(
            'print_iteration_interval', 100)

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        '''
        开始训练(评估)流程
        '''
        print('train_x=',train_x, 'train_y' ,train_y)
        # assert len(train_x) == len(train_y)
        assert len(train_x) == len(self.inputs)

        if test_x is not None and test_y is not None:
            # assert len(test_x) == len(test_y)
            assert len(test_x) == len(self.inputs)

        # 初始化权值变量
        self._variable_weights_init()
        print('[INIT] Variable weights init finished')

        # 传入数据，开始主循环
        self.main_loop(train_x, train_y, test_x, test_y)

    def main_loop(self, train_x, train_y, test_x, test_y):
        '''
        训练（评估）的主循环
        '''

        # 第一层循环，迭代epoches轮
        for self.epoch in range(self.epoches):
            print('start epcho=',self.epoch)
            

            # 模型训练
            self.train(train_x, train_y)

            # 如果需要，对模型进行评估
            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        '''
        使用训练集进行模型训练
        '''
        print('- Epoch [{}] train start, batch size: {}, train data size: {}'.format(
            self.epoch + 1, self.batch_size, len(train_x)))
        #import pdb; pdb.set_trace()
        start_time = time.time()
        last_batch_start_time = time.time()
        last_iter_start_time = time.time()

        # 遍历训练数据集
        for i in range(len(list(train_x.values())[0])):

            # 使用一个样本，执行一次前向传播和反向传播
            self.one_step(self._get_input_values(train_x, i), train_y[i])

            if (i+1) % self.print_iteration_interval == 0:
                print('-- iteration [{}] finished, time cost: {:.2f}  and loss value: {:4f}'.format(
                    i, time.time() - last_iter_start_time, float(self.loss_op.outputs)))
                last_iter_start_time = time.time()

            # 如果次数超过批大小，执行一次更新
            if (i+1) % self.batch_size == 0:
                last_batch_end_time = time.time()
                last_update_start_time = time.time()
                self._optimizer_update()
                computing_cost = last_batch_end_time - last_batch_start_time
                gra_update_cost = time.time() - last_update_start_time
                # print('---- Batch [{}] finished, computing cost: {:.2f}, gradients update cost: {:.2f} and total cost: {:.2f}'.format(
                #     int((i+1)/self.batch_size), computing_cost, gra_update_cost, computing_cost + gra_update_cost))
                last_batch_start_time = time.time()

        print('- Epoch [{}] train finished, time cost: {:.2f}'.format(
            self.epoch + 1, time.time() - start_time))

    def eval(self, test_x, test_y):
        '''
        使用测试集进行模型评估
        '''
        for metrics_op in self.metrics_ops:
            metrics_op.reset()

        for i in range(len(list(test_x.values())[0])):

            self.one_step(self._get_input_values(
                test_x, i), test_y[i], is_eval=True)

            for metrics_op in self.metrics_ops:
                metrics_op.forward()

        metrics_str = 'Epoch [{}] evaluation metrics '.format(self.epoch + 1)
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value_str()

        print(metrics_str)

    def _get_input_values(self, x, index):
        '''
        x是dict类型的数据集，需要取出第index个样本
        '''

        input_values = dict()
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]

        return input_values

    def one_step(self, data_x, data_y, is_eval=False):
        '''
        执行一次前向计算和一次后向计算(可能)
        '''

        for i in range(len(self.inputs)):

            # 根据输入节点的名称，从输入数据dict中找到对应数据
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)

        self.input_y.set_value(np.mat(data_y).T)

        # 只有在训练阶段才执行优化器
        if not is_eval:
            self.optimizer.one_step()

    @abc.abstractmethod
    def _variable_weights_init(self):
        '''
        权值变量初始化，具体的初始化操作由子类完成
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        '''
        调用优化器执行参数更新
        '''
        raise NotImplementedError()

class SimpleTrainer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

    def _variable_weights_init(self):
        '''
        不做统一的初始化操作，使用节点自身的初始化方法
        '''
        pass

    def _optimizer_update(self):
        self.optimizer.update()



class DistTrainerParameterServer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)


    def _variable_weights_init(self):
        '''
        多个worker通过ps保证变量节点初始化一致
        '''
        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Tensor) and node.trainable:
                var_weights_dict[node.name] = node.outputs

        # 把自己的初始值发送给ps，由ps决定使用哪个Worker并返回
        duplicated_var_weights_dict = self.ps_client.variable_weights_init(
            var_weights_dict)

        # 使用ps返回的初始值，重新初始化本地
        for var_name, weights in duplicated_var_weights_dict.items():
            update_node_value_in_graph(var_name, weights)

        print('[INIT] Worker variable weights initialized')


    def _optimizer_update(self):

        # 把当前梯度push到ps上。此操作可能被block，直到所有节点都pull完成
        acc_gradient = self.optimizer.acc_gradient
        self.ps_client.push_gradients(
            acc_gradient, self.optimizer.acc_no)

        # 从ps把所有节点的平均梯度pull回来。此操作可能被block直到所有节点都push完成
        node_gradients_dict = self.ps_client.pull_gradients()

        # 使用平均梯度，利用优化器的优化算法，更新本地变量
        self.optimizer.update(node_gradients_dict)



class DistTrainerRingAllReduce(Trainer):
    '''
    Ring All-Reduce模式的分布式训练
    '''

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

        # 读取集群配置信息和自身信息
        self.cluster_conf = kargs['cluster_conf']
        self.worker_index = kargs['worker_index']

        self.workers = self.cluster_conf['workers']
        self.worker_num = len(self.workers)
        self.host = self.workers[self.worker_index]

        self.step = self.worker_num - 1

        # 根据集群的环状拓扑结构确定右邻居
        self.target_host = self.workers[(
            self.worker_index + 1) % self.worker_num]

        # 本节点是否已被初始化
        self.is_init = False
        self.init_cond = threading.Condition()

        self.cur_partion_index = self.worker_index
        self.partition = []

        # 获取所有可训练节点
        self.variables = get_trainable_variables_from_graph()

        # 根据worker的总数量，对即将更新的变量节点列表进行等长切分
        self._partition_variables()

        # 用于控制梯度的发送和接收
        self.is_recieved = False
        self.recieved_gradients = None
        self.recieved_acc_no = None
        self.cond = threading.Condition()

        # 创建本节点的梯度接收服务
        allreduce.RingAllReduceServer(
            self.host, self.worker_index,
            self._variable_weights_init_callback,
            self._scatter_callback,
            self._gather_callback).serve()

        # 创建连接目标节点的梯度发送client
        self.client = allreduce.RingAllReduceClient(self.target_host)


    def _variable_weights_init(self):

        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Tensor) and node.trainable:
                var_weights_dict[node.name] = node.outputs
        print('[INIT] Send variable init weights to worker ', self.target_host)

        # 第一个节点不需要等待，使用默认值更新给下一个节点
        if self.worker_index == 0:
            self.client.variable_weights_init(var_weights_dict)
        else:
            self.init_cond.acquire()
            while not self.is_init:
                self.init_cond.wait()
            self.init_cond.release()
            self.client.variable_weights_init(var_weights_dict)


    def _variable_weights_init_callback(self, var_weights_dict):

        # 第一个节点不需要接收上一个节点的初始值
        if self.worker_index != 0:
            print('[INIT] Variables initializing weights from last worker node...')
            for var_name, weights in var_weights_dict.items():
                update_node_value_in_graph(var_name, weights)

        # 已初始化完成，通知发送流程
        self.init_cond.acquire()
        self.is_init = True
        self.init_cond.notify_all()
        self.init_cond.release()


    def _optimizer_update(self):

        # 共执行 N-1 次scatter操作，把本worker的梯度切片发送给下一个worker
        # 同时接收左邻居发送过来的梯度，累加到自己的对应切片上
        for scatter_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            cur_acc_no = self.optimizer.acc_no if scatter_index == 0 else self.recieved_acc_no

            # 把自身的一个数据分块发送给右邻居
            self.client.send(gradients_part, cur_acc_no, 'scatter')

            # 等待接收并处理完左邻居节点的数据
            self._wait_for_recieve('scatter')

        # 然后执行 N-1 次all-gather操作，把本worker的梯度切片发送给下一个worker
        # 同时接收上一个worker发送过来的梯度并替换自己的对应切片
        for gather_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            self.client.send(gradients_part, 0, 'gather')
            self._wait_for_recieve('gather')

        self.optimizer.update()


    def _partition_variables(self):
        '''
        根据worker的总数量，对即将更新的权值变量列表进行等长切分
        '''
        var_num = len(self.variables)
        part_length = math.ceil(var_num / self.worker_num)
        assert part_length > 0

        start = 0
        end = start + part_length
        for i in range(self.worker_num - 1):
            self.partition.append((start, end))
            start = end
            end = start + part_length

        self.partition.append((start, var_num))


    def _get_gradients_partition(self):
        '''
        获取下一个梯度切片
        '''
        start, end = self.partition[self.cur_partion_index]
        part_variables = self.variables[start:end]
        self.cur_partion_index = (
            self.cur_partion_index + self.step) % self.worker_num
        part_gradients = dict()
        for var in part_variables:
            part_gradients[var] = self.optimizer.acc_gradient[var]
        return part_gradients


    def _scatter_callback(self, node_gradients_dict, acc_no):
        '''
        Scatter 阶段的回调函数，接收上一个worker发送过来的梯度和样本数
        '''
        if self.cond.acquire():
            while self.is_recieved:
                self.cond.wait()

            # 把接收到的梯度缓存下来
            self.recieved_gradients = node_gradients_dict
            self.recieved_acc_no = acc_no
            self.is_recieved = True

            # 通知主流程，把接收到的梯度更新到优化器
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()


    def _gather_callback(self, node_gradients_dict):
        '''
        All-gather 阶段的回调函数，接收上一个worker发送来的梯度
        '''
        if self.cond.acquire():
            while self.is_recieved:
                self.cond.wait()

            self.recieved_gradients = node_gradients_dict
            self.is_recieved = True

            # 通知主流程，把接收到的梯度更新到优化器
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()


    def _wait_for_recieve(self, stage):
        '''
        等待梯度，并把接收到的梯度更新到优化器中
        '''
        if self.cond.acquire():
            while not self.is_recieved:
                self.cond.wait()

            # 如果是scatter阶段则累加梯度，同时累加样本数
            if stage == 'scatter':
                self.optimizer.apply_gradients(
                    self.recieved_gradients,  summarize=True, acc_no=self.recieved_acc_no)

            # 如果是all-gather阶段则覆盖梯度，样本数保持不变
            else:
                self.optimizer.apply_gradients(
                    self.recieved_gradients, summarize=False, acc_no=self.optimizer.acc_no)

            self.is_recieved = False

            # 梯度已被更新，通知接收流程继续接收新的梯度
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()