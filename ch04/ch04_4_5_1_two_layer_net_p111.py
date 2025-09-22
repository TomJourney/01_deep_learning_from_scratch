import numpy as np
import common.neural_network_active_func as net_func
import common.gradient as grad_func

# 两层神经网络类
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = net_func.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = net_func.softmax_no_overflow(a2)

        return y

    # x:输入特征， t:标签
    def loss(self, x, t):
        y = self.predict(x)
        return net_func.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 返回数组 y 沿着第1轴（axis=1）的最大值的索引（第1轴是列方向）
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入特征， t：标签
    # 计算神经网络损失函数关于权值及偏置的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = grad_func.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = grad_func.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = grad_func.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = grad_func.numerical_gradient(loss_W, self.params['b2'])

        return grads
