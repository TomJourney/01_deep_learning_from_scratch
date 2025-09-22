import numpy as np
import ch04_4_5_1_two_layer_net_p111 as two_layer_network

# 创建两层神经网络实例，其中输入特征784维，隐藏层神经元数100，输出层数据元数10（分类结果）
net = two_layer_network.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape) # (784, 100)
print(net.params['b1'].shape) # (100,)
print(net.params['W2'].shape) # (100, 10)
print(net.params['b2'].shape) # (10,)

print("\n==== params保存了神经网络全部参数，用于推理处理" )
# params保存了神经网络全部参数，用于推理处理
# 生成一个形状为 (100, 784) 的二维数组，其中的元素是从标准正态分布（均值为0，标准差为1）中随机抽取的
x = np.random.randn(100, 784) # 伪输入数据-100笔
y = net.predict(x)

# grads保存了各个参数的梯度
print("\n=== grads保存了各个参数的梯度")
x = np.random.randn(100, 784)  # 伪输入数据-100笔
t = np.random.randn(100, 10)  # 伪正确解标签-100笔
# 计算梯度
grads = net.numerical_gradient(x, t)
# 查看梯度维度
print(grads['W1'].shape) # (784, 100)
print(grads['b1'].shape) # (100,)
print(grads['W2'].shape) # (100, 10)
print(grads['b2'].shape) # (10,)

