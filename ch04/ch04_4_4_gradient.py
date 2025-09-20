import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# 梯度
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成与x形状相同的所有元素都为0的数组

    # 遍历索引
    for index in range(x.size):
        temp_value = x[index]
        # f(x+h)的计算
        x[index] = temp_value +h
        fxh1 = f(x)

        # f(x-h)的计算
        x[index] = temp_value - h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2*h)
        x[index] = temp_value # 还原值
    return grad

