import numpy as np

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

