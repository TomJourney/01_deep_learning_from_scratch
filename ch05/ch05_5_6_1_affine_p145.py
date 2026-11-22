import numpy as np

# 定义输入 X, W, B
X = np.random.rand(2) # 输入
W = np.random.rand(2, 3) # 权重
B = np.random.rand(3) # 偏置

# 打印形状
print("\n========== 打印形状 ==========")
print("X.shape = ", X.shape) # X.shape =  (2,)
print("W.shape = ", W.shape) # W.shape =  (2, 3)
print("B.shape = ", B.shape) # B.shape =  (3,)

# 打印数组元素
print("\n========== 打印数组元素 ==========")
print("X = ", X) # X =  [0.91695592 0.80890395]
print("W = ", W)
# W =  [[0.18874837 0.77728254 0.82428042]
#  [0.33419921 0.63720898 0.04700795]]
print("B = ", B)  # B =  [0.11234063 0.4639297  0.44225022]

# 执行点积运算
print("\n========== 执行点积运算 ==========")
Y = np.dot(X, W) + B
print("Y=", Y) # Y= [0.55574963 1.69210438 1.23610395]

