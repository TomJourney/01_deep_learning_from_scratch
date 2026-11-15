import numpy as np

# ********** 【1】 信号从输入层到第1层的信号传递
print("********** 【1】 信号从输入层到第1层的信号传递\n")

# 使用激活函数sigmoid转换神经元的加权和
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 从输入层到第1层的信号传递
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape) # (2,)
print(B1.shape) # (3,)

# 神经元a1
A1 = np.dot(X, W1) + B1

# 加入 sigmoid 激活函数
Z1 = sigmoid(A1)

print("A1", A1) # A1 [0.3 0.7 1.1]
print("Z1", Z1) # Z1 [0.57444252 0.66818777 0.75026011]


# ********** 【2】 信号从第1层到第2层的信号传递
print("\n**********【2】信号从第1层到第2层的信号传递\n")
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
# (3,)
# (3, 2)
# (2,)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print("A2", A2)
print("Z2", Z2)
# A2 [0.51615984 1.21402696]
# Z2 [0.62624937 0.7710107 ]