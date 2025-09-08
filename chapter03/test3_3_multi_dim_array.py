import numpy as np

# 3.3.1 多维数组
A = np.array([1, 2, 3, 4])
print(A) # [1 2 3 4]
print(np.ndim(A)) # 1  数组的维度
print(A.shape) # (4,)  数组的形状
print(A.shape[0]) # 4

# 二维数组
print("\n=== 二维数组(矩阵)")
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
# [[1 2]
#  [3 4]
#  [5 6]]
print(np.ndim(B)) # 2 数组的维度
print(B.shape) # (3, 2)


# 3.3.2 矩阵乘法
print("\n=== 3.3.2 矩阵乘法")
A= np.array([[1, 2], [3, 4]])
print(A.shape) # (2, 2)

B = np.array([[5, 6], [7, 8]])
print(B.shape) # (2, 2)
print(np.dot(A, B))  # 点积
# [[19 22]
#  [43 50]]

# 2*3矩阵 和 3*2矩阵的乘积
print("\n=== *3矩阵 和 3*2矩阵的乘积")
A= np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape) # (2, 3)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape) # (3, 2)
print(np.dot(A, B))
# [[22 28]
#  [49 64]]

# 3.3.3 神经网络的内积
print("\n=== 3.3.3 神经网络的内积")
X = np.array([1, 2])
print(X.shape) # (2,)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
# [[1 3 5]
#  [2 4 6]]
# (2, 3)

Y = np.dot(X, W)
print(Y)
# [ 5 11 17]







