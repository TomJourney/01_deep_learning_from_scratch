import numpy as np

# 【1】 偏置正向传播时，会把偏置加到每个数据上。
print("\n========== 【1】 偏置正向传播时，会把偏置加到每个数据上。")
X_dot_W = np.array([[0,0,0], [10,10,10]])
B = np.array([1,2,3])

print("\n========== 矩阵加法")
print("\nX_dot_W=", X_dot_W)
# X_dot_W= [[ 0  0  0]
#  [10 10 10]]

print("\nB=", B) # B= [1 2 3]
print("\nX_dot_W + B = ", X_dot_W + B)

# X_dot_W + B =  [[ 1  2  3]
#  [11 12 13]]

# 【2】 反向传播时，各个数据的反向传播的值需要汇总为偏置的元素
print("\n========== 【2】 反向传播时，各个数据的反向传播的值需要汇总为偏置的元素 ")
dY = np.array([[1,2,3], [4,5,6]])
print("\ndY=", dY)
# dY= [[1 2 3]
#  [4 5 6]]

dB = np.sum(dY, axis=0)
print("\ndB=", dB) # dB= [5 7 9]