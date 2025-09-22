import ch04_4_4_2_neural_network_gradient as net_grad
import numpy as np
from common.gradient import numerical_gradient

print(np.random.randn(2, 3))

# 测试案例：计算神经网络的梯度
network = net_grad.simpleNet()
# 打印权重参数
print("\n===打印权重参数")
print(network.W)
# [[-1.01151507  0.29654086  0.54875784]
#  [-0.65483815  1.71663151  0.79564619]]


# 传入输入特征，通过神经网络做预测
x = np.array([0.6, 0.9])
# 通过神经网络计算预测概率
p = network.predict(x)
print("\n=== 通过神经网络计算预测概率")
print(p)
# [-1.19626337  1.72289288  1.04533627]
print("\n=== 计算概率最大值的索引")
print(np.argmax(p))
# 1

print("\n=== 计算损失函数")
t = np.array([0, 0, 1])  # 正确解标签
print(network.loss(x, t))
# 1.1234180817699424


##  计算损失函数关于权值的梯度
print("\n=== 权值, network.W = ")
print(network.W)
# [[-0.78391423 -0.0493546   1.01982955]
#  [-0.1693144  -0.82591994 -0.70009164]]

print("\n=== 计算损失函数关于权值的梯度")
def busi_func(W):
    return network.loss(x,t)
dW = numerical_gradient(busi_func, network.W)
print(dW)
# [[ 0.16255973  0.13988719 -0.30244691]
#  [ 0.24383959  0.20983078 -0.45367037]]

