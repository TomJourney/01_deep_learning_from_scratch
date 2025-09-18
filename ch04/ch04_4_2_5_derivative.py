import sys, os
import numpy as np


# 导数的不好实现例子
def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h


# 测试案例
def function_1(x):
    return x ** 2


# 计算导数
print(function_1(2 + 10e-50))  # 4.0
print(function_1(2))  # 4.0
derivative = numerical_diff(function_1, 2)
print(derivative)  # 0 错误结果 （准确结果应该是4）

# 因为默认情况下，python没有办法精确表示 10e-50，而是用0来近似
print("\n\n=== np.float32 ")
print(np.float32(1e-50))  # 0.0
print(np.float64(1e-50))  # 1e-50
