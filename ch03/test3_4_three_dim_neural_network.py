import numpy as np

# 3.4 三层神经网络的实现
# 3.4.2 各层间信号传递的实现
print("\n=== 3.4.2 各层间信号传递的实现")
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape) # (2,)
print(B1.shape) # (3,)

# 第1层神经网络各个神经元的加权和
A1 = np.dot(X, W1) + B1
print(A1) # [0.3 0.7 1.1]

# 使用激活函数sigmoid转换神经元的加权和
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("\n=== 激活函数转换各神经元的加权和")
Z1 = sigmoid(A1)
print(Z1) # [0.57444252 0.66818777 0.75026011]


print("\n=== 第1层到第2层的信号传递")
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)
# (3,)
# (3, 2)
# (2,)

# 第1层神经元到第2层神经元的信号传递
A2 = np.dot(Z1, W2) + B2
print(A2) # [0.51615984 1.21402696]
Z2 = sigmoid(A2)
print(Z2) # [0.62624937 0.7710107 ]


print("\n=== 第2层到输出层的信号传递")
# identy_func : 是输出层的激活函数，也称恒等函数
def identity_func(x):
    return x
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_func(A3)



