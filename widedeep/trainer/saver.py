import sys
sys.path.append('..')

import numpy as np
#import widedeep as ms
from node.node import *
import json
from graph.graph import default_graph
import os
import datetime

class ClassMining(object):
    @classmethod
    def get_subclass_list(cls, model):
        subclass_list = []
        for subclass in model.__subclasses__():
            subclass_list.append(subclass)
            subclass_list.extend(cls.get_subclass_list(subclass))
        return subclass_list

    @classmethod
    def get_subclass_dict(cls, model):
        subclass_list = cls.get_subclass_list(model=model)
        return {k: k.__name__ for k in subclass_list}

    @classmethod
    def get_subclass_names(cls, model):
        subclass_list = cls.get_subclass_list(model=model)
        return [k.__name__ for k in subclass_list]

    @classmethod
    def get_instance_by_subclass_name(cls, model, name):
        for subclass in model.__subclasses__():
            if subclass.__name__ == name:
                return subclass
            instance = cls.get_instance_by_subclass_name(subclass, name)
            if instance:
                return instance


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


class Saver():

    def __init__(self, root_dir = '.'):

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)


    def save(self, model_file_name='model.json', weights_file_name='weights.npz'):
        self._save_model_and_weights(default_graph, model_file_name, weights_file_name)


    def _save_model_and_weights(self, graph, model_file_name, weights_file_name):

        model_json = {
            #'meta': meta,
            #'service': service
        }
        
        graph_json = []
        weights_dict = dict()

        # 把节点元信息保存为dict/json格式
        for node in graph.nodes:
            '''if not node.need_save:
                continue'''
            print('Saving node: {}'.format(node.name))
            node.kargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.childs],
                'kargs': node.kargs
            }

            # 保存节点的dim信息
            if node.outputs is not None:
                if isinstance(node.outputs, np.matrix):
                    node_json['dim'] = node.outputs.shape

            graph_json.append(node_json)

            # 如果节点是Variable类型，保存其值
            # 其他类型的节点不需要保存
            if isinstance(node, Tensor):
                weights_dict[node.name] = node.outputs

        model_json['graph'] = graph_json

        # json格式保存计算图元信息
        model_file_path = os.path.join(self.root_dir, model_file_name)

        print('======model_file_path', model_file_path)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4)
            print('Save model into file: {}'.format(model_file.name))

        # npz格式保存节点值（Variable节点）
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'wb') as weights_file:
            print(weights_file, weights_dict)
            np.savez(weights_file, **weights_dict)
            print('Save weights to file: {}'.format(weights_file.name))


    


    @staticmethod
    def create_node(graph, from_model_json, node_json):
        '''
        静态工具函数，递归创建不存在的节点
        '''
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dim = node_json.get('dim', None)
        kargs = node_json.get('kargs', None)
        kargs['graph'] = graph

        parents = []
        for parent_name in parents_name:
            parent_node = get_node_from_graph(parent_name, graph=graph)
            if parent_node is None:
                parent_node_json = None
                for node in from_model_json:
                    if node['name'] == parent_name:
                        parent_node_json = node

                assert parent_node_json is not None
                # 如果父节点不存在，递归调用
                parent_node = Saver.create_node(
                    graph, from_model_json, parent_node_json)

            parents.append(parent_node)

        # 反射创建节点实例
        print('!!!!!!!!!!!!!!!!!',node_name, node_type, parents, dim, kargs)
        if node_type == 'Tensor':
            assert dim is not None

            dim = tuple(dim)
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, shape=dim, name=node_name, **kargs)
        else:
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, name=node_name, **kargs)
    
    def _restore_nodes(self, graph, from_model_json, from_weights_dict):

        for index in range(len(from_model_json)):
            node_json = from_model_json[index]
            node_name = node_json['name']

            weights = None
            if node_name in from_weights_dict:
                weights = from_weights_dict[node_name]
                print('Restore node {} with weights'.format(node_name), weights)

            # 判断是否创建了当前节点，如果已存在，更新其权值
            # 否则，创建节点
            #print('from_model_json', from_model_json)
            print('node_json', node_json)
            target_node = get_node_from_graph(node_name, graph=graph)
            if target_node is None:
                print('Target node {} of type {} not exists, try to create the instance'.format(
                    node_json['name'], node_json['node_type']))
                print('from_model_json', from_model_json)
                print('node_json', node_json)
                target_node = Saver.create_node(
                    graph, from_model_json, node_json)
            else:
                print('target_node {} is not none', target_node)

            target_node.outputs = weights

    
    def load(self, to_graph=None,
             model_file_name= '/model.json',
             weights_file_name= '/weights.npz'):
        '''
        从文件中读取并恢复计算图结构和相应的值
        '''
        if to_graph is None:
            to_graph = default_graph

        model_json = {}
        graph_json = []
        weights_dict = dict()

        # 读取计算图结构元数据
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)

        # 读取计算图节点值数据
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz_files = np.load(weights_file)
            for file_name in weights_npz_files.files:
                weights_dict[file_name] = weights_npz_files[file_name]
            weights_npz_files.close()

        graph_json = model_json['graph']
        self._restore_nodes(to_graph, graph_json, weights_dict)
        print('Load and restore model from {} and {}'.format(
            model_file_path, weights_file_path))

        self.meta = model_json.get('meta', None)
        self.service = model_json.get('service', None)
        return self.meta, self.service

        



    