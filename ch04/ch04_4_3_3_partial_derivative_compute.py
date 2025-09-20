import numpy as np
import matplotlib.pyplot as plt
import ch04_4_2_5_derivative_v2 as devivativeV2

# 计算偏导数
# 计算 f(x0,x1) 在x0=3，x1=4的偏导数
# f(x0,x1) = x0^2 + x1 ^2

# 计算 f(x0,x1) 在x0=3的偏导数，固定x1=4
def func_1(x0):
    return x0**2 + 4**2
print(devivativeV2.numerical_diff_v2(func_1, 3.0))
# 6.00000000000378

# 计算 f(x0,x1) 在x1=4的偏导数，固定x0=3
def func_2(x1):
    return 3.0**2 + x1**2

print(devivativeV2.numerical_diff_v2(func_2, 4.0)) # 7.999999999999119
