import sys, os

from ch04.ch04_4_2_2_cross_entropy_error import cross_entropy_error
from common.neural_network_active_func import softmax_no_overflow
sys.path.append(os.pardir)
import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 用高斯分布进行初始化

    # 预测方法
    def predict(self, x):
        return np.dot(x, self.W)

    # 计算损失， 其中x是输入特征，t是正确解标签
    def loss(self, x, t):
        z = self.predict(x) # x点乘权重，得到预测概率数组
        y = softmax_no_overflow(z) # 通过softmax函数找出概率最大索引
        loss = cross_entropy_error(y, t) # 计算交叉熵损失函数（y是预测的索引，t是测试标签或正确解标签）
        return loss
