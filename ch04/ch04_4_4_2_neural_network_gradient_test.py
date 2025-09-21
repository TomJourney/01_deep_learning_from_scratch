import ch04_4_4_2_neural_network_gradient as net_grad
import numpy as np

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

print("\\n=== 计算损失函数")
t = np.array([0, 0, 1])
print(network.loss(x, t))
# 1.1234180817699424

