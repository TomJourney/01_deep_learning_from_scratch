# Softmax-with-Loss层的代码实现

from common.neural_network_active_func import sigmoid, softmax_no_overflow
from ch04.ch04_4_2_2_cross_entropy_error import cross_entropy_error

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax输出
        self.t = None # 监督数据 (ont-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax_no_overflow(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
