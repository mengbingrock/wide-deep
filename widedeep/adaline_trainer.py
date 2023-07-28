import sys
sys.path.append('.')


import numpy as np
#import widedeep as ms


#from widedeep.node import *


import argparse
from trainer.trainer import SimpleTrainer


import numpy as np
from node.node import Tensor
from node.ops import *
from node.loss import *

from trainer.optimizer import *
from trainer.saver import Saver
from graph.graph import (default_graph, get_node_from_graph)
from node.metrics import *


"""
制造训练样本。根据均值171，标准差6的正态分布采样500个男性身高，根据均值158，
标准差5的正态分布采样500个女性身高。根据均值70，标准差10的正态分布采样500个
男性体重，根据均值57，标准差8的正态分布采样500个女性体重。根据均值16，标准差
2的正态分布采样500个男性体脂率，根据均值22，标准差2的正态分布采样500个女性体
脂率。构造500个1，作为男性标签，构造500个-1，作为女性标签。将数据组装成一个
1000 x 4的numpy数组，前3列分别是身高、体重和体脂率，最后一列是性别标签。
"""

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

# 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
x = Tensor(shape=(3, 1), init=False, trainable=False)

# 类别标签，1男，-1女
label = Tensor(shape=(1, 1), init=False, trainable=False)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w = Tensor(shape=(1, 3), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b = Tensor(shape=(1, 1), init=True, trainable=True)

# ADALINE的预测输出

output = Add(MatMul(w, x), b)
predict = Step(output)

# 损失函数
loss = PerceptionLoss(MatMul(label, output))

# 学习率
learning_rate = 0.0001

# 优化器
optimizer = GradientDescent(default_graph, loss, learning_rate)

accuracy = Accuracy(output, label)

# 使用PS训练器，传入集群配置信息
trainer = SimpleTrainer([x], label, loss, optimizer,
                                        epoches=10, batch_size=10,
                                        eval_on_train=True, metrics_ops=[accuracy],
                                        )
print('x.name=',x.name, 'train_set[:,:-1].shape=',train_set[:,:-1].shape, 'train_set[:,-1].shape=',train_set[:,-1].shape)

trainer.train_and_eval({x.name: train_set[:,:-1]}, train_set[:, -1],
                        {x.name: train_set[-10:-1,:-1]}, train_set[-10:-1, -1])

#exporter = Exporter()
#sig = exporter.signature('img_input', 'softmax_output')

saver = Saver('.')
saver.save(model_file_name='trainer_model.json', weights_file_name='trainer_weights.npz')
    

# 每个epoch结束后评价模型的正确率
pred = []

# 遍历训练集，计算当前模型对每个样本的预测值
for i in range(len(train_set)):
#for i in range(1):

    features = np.mat(train_set[i, :-1]).T
    x.set_value(features)

    # 在模型的predict节点上执行前向传播
    predict.forward()
    pred.append(predict.outputs[0, 0])  # 模型的预测结果：1男，0女

pred = np.array(pred) * 2 - 1  # 将1/0结果转化成1/-1结果，好与训练标签的约定一致
#print('======pred:', pred)
# 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)

# 打印当前epoch数和模型在训练集上的正确率


# 保存模型
saver = Saver('.')

saver.save()

saver.load(model_file_name='trainer_model.json', weights_file_name='trainer_weights.npz')

x = get_node_from_graph("Tensor:0")
pred = get_node_from_graph("Step:6")

print(x, pred)
preds = []

for i in range(100):
    #for i in range(1):

    features = np.mat(train_set[i, :-1]).T
    #print('features=',features)
    #print("===features.shape, value =",features.shape, features)
    x.set_value(features)

    # 在模型的predict节点上执行前向传播
    pred.forward()

    print('outputs, pred', output.outputs, pred.outputs[0, 0])

    preds.append(pred.outputs[0, 0])


preds = np.array(preds) * 2 - 1  # 将1/0结果转化成1/-1结果，好与训练标签的约定一致
#print('======pred:', pred)
# 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
accuracy = (train_set[:100, -1] == preds).astype(np.int).sum() / 100

# 打印当前epoch数和模型在训练集上的正确率
print("accuracy: {:.3f}".format(accuracy))
