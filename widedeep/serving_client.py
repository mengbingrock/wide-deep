
import sys
sys.path.append('.')

import grpc
import numpy as np

#from . import *
from serving import serving_pb2, serving_pb2_grpc



class MatrixSlowServingClient(object):
    def __init__(self, host):
        self.stub = serving_pb2_grpc.MatrixSlowServingStub(
            grpc.insecure_channel(host))
        print('[GRPC] Connected to MatrixSlow serving: {}'.format(host))

    def Predict(self, mat_data_list):
        req = serving_pb2.PredictReq()
        for mat in mat_data_list:
            proto_mat = req.data.add()
            proto_mat.outputs.extend(np.array(mat).flatten())
            proto_mat.shape.extend(list(mat.shape))

        resp = self.stub.Predict(req)
        return resp


if __name__ == '__main__':
    # 加载MNIST数据集，取一部分样本并归一化
    male_heights = np.random.normal(171, 6, 500)
    female_heights = np.random.normal(158, 5, 500)

    male_weights = np.random.normal(70, 10, 500)
    female_weights = np.random.normal(57, 8, 500)

    male_bfrs = np.random.normal(16, 2, 500)
    female_bfrs = np.random.normal(22, 2, 500)

    male_labels = [1] * 500
    female_labels = [-1] * 500

    train_set = np.array([np.concatenate((male_heights, female_heights)),
                        np.concatenate((male_weights, female_weights)),
                        np.concatenate((male_bfrs, female_bfrs)),
                        np.concatenate((male_labels, female_labels))]).T

    # 随机打乱样本顺序
    np.random.shuffle(train_set)

    test_data = train_set[:10]

    

    host = '127.0.0.1:5000'
    client = MatrixSlowServingClient(host)

    
    #exit()
    for index in range(len(test_data)):
        x = np.mat(train_set[index, :-1]).T
        label = train_set[index, -1]
        
        resp = client.Predict([x])
        
        resp_mat_list = []
        for proto_mat in resp.data:
            dim = tuple(proto_mat.shape)
            mat = np.mat(proto_mat.outputs, dtype=np.float32)
            mat = np.reshape(mat, dim)
            resp_mat_list.append(mat)
        pred = resp_mat_list[0].data[0,0] * 2 - 1
        
        gt = label
        print('model predict {} and ground truth: {}'.format(
            pred, gt))