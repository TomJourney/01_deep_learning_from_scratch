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
        z = self.predict(x) # x点乘权重，得到权重累加值
        # Softmax函数是一种将任意实数向量转化为概率分布的归一化指数函数，其输出向量的每个元素都在0到1之间，且所有元素的和为1
        y = softmax_no_overflow(z) #
        loss = cross_entropy_error(y, t) # 计算交叉熵损失函数（y是预测的索引，t是测试标签或正确解标签）
        return loss
