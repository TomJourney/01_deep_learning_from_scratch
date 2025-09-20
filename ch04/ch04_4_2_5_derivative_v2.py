import numpy as np
import matplotlib.pyplot as plt


# 基于中心差分的数值梯度(数值微分函数)
def numerical_diff_v2(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试案例
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x
